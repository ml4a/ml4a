from tqdm import tqdm
import random
import pickle
import imageio
import scipy
import numpy as np
import torch


# TODO
# labels and label interpolation




from ..utils import downloads
from .. import image
from . import submodules

with submodules.import_from('stylegan2-ada-pytorch'):  # localimport fails here
    import dnnlib
    import torch_utils


G = None

noise_modes = ['const', 'random', 'none']

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
        
        
def generate(latents, labels=None, truncation=1.0, noise_mode='const'):
    assert noise_mode in noise_modes, \
        'Error: noise mode %s not found. Available are %s' % (noise_mode, ', '.join(noise_modes))
    if isinstance(latents, np.ndarray):
        latents = torch.from_numpy(latents).cuda()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).cuda()
    img = G(latents, labels, truncation_psi=truncation, noise_mode=noise_mode)
    img = (img.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img


def random_sample(num_images, label=None, truncation=1.0, seed=None):
    seed = seed if seed else np.random.randint(100)
    rnd = np.random.RandomState(int(seed))
    latents = rnd.randn(num_images, G.z_dim)
    
    ### fix labels
    labels = None #np.zeros((num_images, 7))
    return generate(latents, labels, truncation)







### fix labels

def interpolated_matrix_between(start, end, num_frames):
    linfit = interp1d([0, num_frames-1], np.vstack([start, end]), axis=0)
    interp_matrix = np.zeros((num_frames, start.shape[1]))
    for f in range(num_frames):
        interp_matrix[f, :] = linfit(f)
    return interp_matrix

def get_interpolated_labels(labels, num_frames=60):
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

def get_latent_interpolation(endpoints, num_frames_per, mode, shuffle):
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
            r = 0.5 - 0.5 * np.cos(np.pi*t/(num_frames_per-1)) if mode == 'ease' else float(t) / num_frames_per
            latents[frame, :] = (1.0-r) * endpoints[e1,:] + r * endpoints[e2,:]
    return latents

def get_latent_interpolation_bspline(endpoints, nf, k, s, shuffle):
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
    
    assert G is not None, 'Error: no mode loaded'
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    random_state = np.random.RandomState(seed if seed is not None else np.random.randint(1000))
    shape = [num_frames, np.prod([1, 1])] + [G.z_dim]  # [frame, image, channel, component]
    latents = random_state.randn(*shape).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [smoothing_sec * mp4_fps] + [0]*2, mode='wrap')
    latents /= np.sqrt(np.mean(np.square(latents)))
    return latents


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
    
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    all_latents = get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps, seed)

    
    all_labels = get_interpolated_labels(labels, num_frames)  ###### WRONG
    
    
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
