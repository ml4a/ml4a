from tqdm import tqdm
import re
import os
import sys
import json
import subprocess
import copy
import random
import pickle
import numba
import imageio
import scipy
import numpy as np
import torch
import torch.nn.functional as F
import PIL

from ..utils import downloads, EasyDict
from .. import image
from . import submodules

with submodules.import_from('stylegan2-ada-pytorch'):  # localimport fails here
    import dnnlib
    import torch_utils
    import legacy
    import projector

G = None

noise_modes = ['const', 'random', 'none']
preset_base_configs = ['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']
resume_presets = ['ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256']
dataset_transforms = [None, 'center-crop', 'center-crop-wide']

pretrained_models = {
    'afhqcat': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/afhqcat.pkl'
    },
    'afhqdog': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/afhqdog.pkl'
    },
    'afhqwild': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/afhqwild.pkl'
    },
    'brecahad': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/brecahad.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/brecahad.pkl'
    },
    'cifar10': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/cifar10.pkl'
    },
    'ffhq': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    },
    'metfaces': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl', 
        'output_path': 'stylegan2-ada-pytorch/pretrained/metfaces.pkl'
    }
}
    

def get_pretrained_models():
    return pretrained_models.keys()


def get_pretrained_model(model_name):
    if model_name in pretrained_models:
        model = pretrained_models[model_name]
        model_file = downloads.download_data_file(
            url=model['url'], 
            output_path=model['output_path'])
        return model_file
    else:
        raise Exception("No pretrained model named %s. Run stylegan.get_pretrained_models() to get a list." % model_name) 


def load_model(network_pkl, randomize_noise=False):
    global G
    with open(network_pkl, 'rb') as f:   
        G = pickle.load(f)['G_ema'].cuda()


def generate(latents, 
             labels=None, 
             truncation=1.0, 
             noise_mode='const',
             minibatch_size=16):
    
    assert G is not None, \
        'Error: no model loaded'
    assert noise_mode in noise_modes, \
        'Error: noise mode %s not found. Available are %s' % (noise_mode, ', '.join(noise_modes))

    if isinstance(latents, np.ndarray):
        latents = torch.from_numpy(latents).cuda()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).cuda()

    # generate image
    if latents.shape[-2] == 18:
        imgs = G.synthesis(latents, noise_mode=noise_mode)
    else:
        imgs = G(latents, labels, truncation_psi=truncation, noise_mode=noise_mode)
    
    # post-processing
    imgs = (imgs.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgs = [PIL.Image.fromarray(img.numpy()) for img in imgs]
    return imgs


def random_sample(num_images, 
                  label=None, 
                  truncation=1.0, 
                  seed=None):
    
    seed = seed if seed else np.random.randint(100)
    rnd = np.random.RandomState(int(seed))
    latents = rnd.randn(num_images, G.z_dim)
    
    ### fix labels
    labels = None #np.zeros((num_images, 7))
    return generate(latents, labels, truncation)


def interpolated_matrix_between(start, 
                                end, 
                                num_frames):
    
    linfit = interp1d([0, num_frames-1], np.vstack([start, end]), axis=0)
    interp_matrix = np.zeros((num_frames, start.shape[1]))
    for f in range(num_frames):
        interp_matrix[f, :] = linfit(f)
    return interp_matrix


def get_interpolated_labels(labels, 
                            num_frames=60):
    
    all_labels = np.zeros((num_frames, 7))
    if type(labels) == list:
        num_labels = len(labels)
        for l in range(num_labels-1):
            e1, e2 = int(num_frames * l / (num_labels-1)), int(num_frames * (l+1) / (num_labels-1))
            start, end = np.zeros((1, 7)), np.zeros((1, 7))
            start[:, labels[l]] = 1
            end[:, labels[l+1]] = 1
            all_labels[e1:e2, :] = interpolated_matrix_between(start, end, e2-e1)
    else:
        all_labels[:, labels] = 1
    return all_labels


def get_latent_interpolation(endpoints, 
                             num_frames_per, 
                             mode='normal', 
                             shuffle=False):
    
    if shuffle:
        random.shuffle(endpoints)
    num_endpoints, dim = len(endpoints), len(endpoints[0])
    num_frames = num_frames_per * num_endpoints
    endpoints = np.array(endpoints)
    latents = np.zeros((num_frames, dim))
    for e in range(num_endpoints):
        e1, e2 = e, (e+1) % num_endpoints
        for t in range(num_frames_per):
            frame = e * num_frames_per + t
            if mode == 'ease':
                r = 0.5 - 0.5 * np.cos(np.pi*t/(num_frames_per-1))
            else:
                r = float(t) / num_frames_per
            latents[frame, :] = (1.0-r) * endpoints[e1,:] + r * endpoints[e2,:]
    return latents


def get_latent_interpolation_bspline(endpoints, 
                                     nf, 
                                     k, 
                                     s, 
                                     shuffle):
    
    if shuffle:
        random.shuffle(endpoints)
    x = np.array(endpoints)
    x = np.append(x, x[0,:].reshape(1, x.shape[1]), axis=0)
    nd = x.shape[1]
    latents = np.zeros((nd, nf))
    nss = list(range(1, 10)) + [10]*(nd-19) + list(range(10,0,-1))
    for i in tqdm(range(nd-9)):
        idx = list(range(i,i+10))
        tck, u = interpolate.splprep([x[:,j] for j in range(i,i+10)], k=k, s=s)
        out = interpolate.splev(np.linspace(0, 1, num=nf, endpoint=True), tck)
        latents[i:i+10,:] += np.array(out)
    latents = latents / np.array(nss).reshape((512, 1))
    return latents.T


def get_gaussian_latents(duration_sec, 
                         smoothing_sec, 
                         mp4_fps=30, 
                         seed=None):
    
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    random_state = np.random.RandomState(seed if seed is not None else np.random.randint(1000))
    shape = [num_frames, np.prod([1, 1])] + [G.z_dim]  # [frame, image, channel, component]
    latents = random_state.randn(*shape).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [smoothing_sec * mp4_fps] + [0]*2, mode='wrap')
    latents /= np.sqrt(np.mean(np.square(latents)))
    return latents



def generate_video(output_path,
                   latents, 
                   labels, 
                   truncation=1.0, 
                   noise_mode='const',
                   mp4_fps=30, 
                   mp4_codec='libx264', 
                   mp4_bitrate='16M', 
                   minibatch_size=16):
    
    assert G is not None, 'Error: no model loaded'
    num_frames = latents.shape[0]
    
    video = imageio.get_writer(output_path, mode='I', fps=mp4_fps, 
                               codec=mp4_codec, bitrate=mp4_bitrate)    

    for f in tqdm(range(0, num_frames, minibatch_size)):
        f1, f2 = f, min(num_frames, f + minibatch_size - 1)
        latents_ = latents.squeeze()[f1:f2]
        labels_ = labels.squeeze()[f1:f2] if labels is not None else None
        images = generate(latents_, labels_, truncation=truncation, noise_mode=noise_mode)
        for image in images:
            video.append_data(np.array(image))
    
    video.close()
    return output_path


def generate_interpolation_video(output_path, 
                                 labels, 
                                 truncation=1.0, 
                                 noise_mode='const',
                                 duration_sec=60.0, 
                                 smoothing_sec=1.0, 
                                 mp4_fps=30, 
                                 mp4_codec='libx264', 
                                 mp4_bitrate='16M', 
                                 seed=None, 
                                 minibatch_size=16):
    
    assert G is not None, 'Error: no model loaded'
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    all_latents = get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps, seed)

    # fix this...
    all_labels = get_interpolated_labels(labels, num_frames) 
    
    
    generate_video(output_path,
                   all_latents, 
                   all_labels, 
                   truncation=1.0,
                   noise_mode='const',
                   mp4_fps=30, 
                   mp4_codec='libx264', 
                   mp4_bitrate='16M', 
                   minibatch_size=16)
    
    video = imageio.get_writer(output_path, mode='I', fps=mp4_fps, 
                               codec=mp4_codec, bitrate=mp4_bitrate)    

    for f in tqdm(range(0, num_frames, minibatch_size)):
        f1, f2 = f, min(num_frames, f + minibatch_size - 1)
        latents = all_latents.squeeze()[f1:f2]
        labels = all_labels.squeeze()[f1:f2]
        images = generate(latents, labels, truncation=truncation, noise_mode=noise_mode)
        for image in images:
            video.append_data(np.array(image))
    
    video.close()
    return output_path


def encode(target_img,
           num_steps=1000,
           w_avg_samples=10000,
           initial_learning_rate=0.1,
           initial_noise_factor=0.05,
           lr_rampdown_length=0.25,
           lr_rampup_length=0.05,
           noise_ramp_length=0.75,
           regularize_noise_weight=1e5,
           device='cuda',
           verbose=True):

    global G
    assert G is not None, 'Error: no model loaded'
    
    # Load target image
    if isinstance(target_img, np.ndarray):
        target_img = PIL.Image.fromarray(target_img.astype(np.uint8))
    w, h = target_img.size
    s = min(w, h)
    target_img = target_img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_img = target_img.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_img = np.array(target_img, dtype=np.uint8)
    target_img = torch.tensor(target_img.transpose([2, 0, 1]), device=device)

    # copy model
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # run projector
    projected_w_steps = projector.project(
        G,
        target=target_img,
        num_steps=num_steps,
        w_avg_samples=w_avg_samples,
        initial_learning_rate=initial_learning_rate,
        initial_noise_factor=initial_noise_factor,
        lr_rampdown_length=lr_rampdown_length,
        lr_rampup_length=lr_rampup_length,
        noise_ramp_length=noise_ramp_length,
        regularize_noise_weight=regularize_noise_weight,
        device=device,
        verbose=verbose
    )
    
    # return final step
    projected_w = projected_w_steps[-1].unsqueeze(0)
    return projected_w


def make_dataset_label_lookup(images_folder):
    folders = [os.path.relpath(f[0], images_folder) 
               for f in os.walk(images_folder)][1:]

    assert len(folders), "Error: labels set True, but no subfolders found"

    file_labels = {'labels': []}
    for class_idx, folder in enumerate(folders):
        class_files = [filename for filename in os.listdir(os.path.join(images_folder, folder))]
        class_files = [[os.path.join(folder, f), class_idx] for f in class_files 
                       if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
        file_labels['labels'] += class_files

    output_file = os.path.join(images_folder, 'dataset.json')
    with open(output_file, 'w') as outfile:
        json.dump(file_labels, outfile)


def dataset_tool(config):
                 
    assert 'images_folder' in config
    assert 'dataset_output' in config

    # unpack configuration
    cfg = EasyDict(config)
    images_folder = cfg.images_folder if 'images_folder' in cfg else 512
    dataset_output = cfg.dataset_output if 'dataset_output' in cfg else 512
    labels = cfg.labels if 'labels' in cfg else 512
    transform = cfg.transform if 'transform' in cfg else None
    size = cfg.size if 'size' in cfg else 512

    assert transform in dataset_transforms, \
        'Transform {} not found, available are: center-crop, center-crop-wide, None'.format(cfg.transform)

    dataset_tool = os.path.join(
        os.path.dirname(os.path.abspath(submodules.__file__)), 
        'stylegan2-ada-pytorch/dataset_tool.py')
        
    popen_args = [
        'python', dataset_tool,
        '--source', '{}'.format(images_folder.replace(" ", "\ ")), 
        '--dest', '{}'.format(dataset_output.replace(" ", "\ ")),
        '--transform', transform,
        '--width', str(size), 
        '--height', str(size)
    ]
    print(' '.join(popen_args))

    if labels:
        make_dataset_label_lookup(images_folder)
    
    process = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''): 
        print(c.decode("utf-8", "ignore") , end='')
    for c in iter(lambda: process.stderr.read(1), b''): 
        print(c.decode("utf-8", "ignore") , end='')

    print('completed dataset pre-processing in {}'.format(dataset_output))


def train(config):
    assert 'results_dir' in config
    assert 'dataset_root' in config

    # unpack configuration
    cfg = EasyDict(config)
    results_dir = cfg.results_dir
    dataset_root = cfg.dataset_root
    save_every = cfg.save_every if 'save_every' in cfg else 50
    mirror = cfg.mirror if 'mirror' in cfg else False
    labels = cfg.labels if 'labels' in cfg else False 
    base_config = cfg.base_config if 'base_config' in cfg else 'auto'
    kimg = cfg.kimg if 'kimg' in cfg else 25000
    resume = cfg.resume if 'resume' in cfg else None
    gpu = cfg.gpu if 'gpu' in cfg else None

    assert cfg.base_config in preset_base_configs, \
        'Base config {} not found, available are: '.format(cfg.base_config, ', '.join(preset_base_configs))

    # determine which GPUs to use
    if gpu is None:
        num_gpus = len(numba.cuda.gpus)
        gpus = ','.join([str(g) for g in range(num_gpus)])
    else:
        gpu = gpu if isinstance(gpu, list) else [gpu]
        num_gpus = len(gpu)
        gpus = ','.join([str(g) for g in gpu])
        
    # don't use 3 gpu, only 1, 2 or 4
    num_gpus = 2 if num_gpus==3 else num_gpus
    
    # run training script
    training_script = os.path.join(
        os.path.dirname(os.path.abspath(submodules.__file__)), 
        'stylegan2-ada-pytorch/train.py'
    )
    
    # if resume set to auto, find last saved checkpoint automagically
    if resume == 'auto':
        dataset_name = os.path.split(dataset_root)[-1]
        checkpoints = [x[0] for x in os.walk(results_dir)][1:]
        regex = r'[0-9]+-{}-{}-.+'.format(dataset_name, base_config)
        matches = [re.findall(regex, c) for c in checkpoints]
        pkls = [glob.glob('{}/{}/network-snapshot-*.pkl'.format(results_dir, m[0])) for m in matches]
        pkls = sorted([item for sublist in pkls for item in sublist])  # flatten
        last_pkl = pkls[-1]
        matches = sorted([m[0] for m in matches if len(m)])
        print("found matches", matches)
        # what if most recent folder has no checkpoint?
        match = matches[-1] if len(matches) else None
        if match:
            checkpoint_dir = os.path.join(results_dir, match)
            files = [f for f in os.listdir(checkpoint_dir) 
                     if os.path.isfile(os.path.join(checkpoint_dir, f))]
            checkpoints = [f for f in files if 'network-snapshot' in f]
            checkpoints = sorted(checkpoints)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            print("check path", resume)
            resume = checkpoint_path
    
    # setup command
    popen_args = [
        'python', training_script,
        '--outdir', results_dir.replace(" ", "\ "),
        '--data', dataset_root.replace(" ", "\ "),
        '--cond', '1' if labels else '0',
        '--snap', str(save_every),
        '--mirror', '1' if mirror else '0',
        '--cfg', base_config,
        '--kimg', str(kimg),
        '--resume', resume,
        '--gpus', str(num_gpus)
    ]
    
    print(' '.join(popen_args))

    # run command
    if gpu is not None:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = gpus
        process = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env)
    else:
        process = subprocess.Popen(popen_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    # log output
    for c in iter(lambda: process.stdout.read(1), b''): 
        print(c.decode("utf-8") , end='')
    for c in iter(lambda: process.stderr.read(1), b''): 
        print(c.decode("utf-8") , end='')
            
    # finished
    print('completed training at {}'.format(results_dir))
