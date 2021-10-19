import os
import random
import math
import PIL.Image
import moviepy.editor
import matplotlib.pyplot as pyplot
import scipy
from scipy.interpolate import interp1d
import numpy as np
import torch

from ..utils import downloads
from .. import image
from . import submodules

# todo: absorb latent code to ml4a.utils.latents

cuda_available = submodules.cuda_available()
device = torch.device('cuda')

#with submodules.localimport('submodules/stylegan3') as _importer:
with submodules.import_from('stylegan3'):  # localimport fails here
    import dnnlib
    import legacy
    import gen_images
#    import pretrained_networks
    #import dnnlib.tflib as tflib


    
G = None

base_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'

pretrained_models = {
    'ffhq-t': {
        'url': '{}/stylegan3-t-ffhq-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/ffhq/stylegan3-t-ffhq-1024x1024.pkl'
    },
    'ffhq-r': {
        'url': '{}/stylegan3-r-ffhq-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/ffhq/stylegan3-r-ffhq-1024x1024.pkl'
    },
    'ffhqu-t': {
        'url': '{}/stylegan3-t-ffhqu-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/ffhqu/stylegan3-t-ffhqu-1024x1024.pkl'
    },
    'ffhqu-r': {
        'url': '{}/stylegan3-r-ffhqu-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/ffhqu/stylegan3-r-ffhqu-1024x1024.pkl'
    },
    'metfaces-t': {
        'url': '{}/stylegan3-t-metfaces-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/metfaces/stylegan3-t-metfaces-1024x1024.pkl'
    },
    'metfaces-r': {
        'url': '{}/stylegan3-r-metfaces-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/metfaces/stylegan3-r-metfaces-1024x1024.pkl'
    },
    'metfaces-t': {
        'url': '{}/stylegan3-t-metfacesu-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/metfacesu/stylegan3-t-metfacesu-1024x1024.pkl'
    },
    'metfaces-r': {
        'url': '{}/stylegan3-r-metfacesu-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/metfacesu/stylegan3-r-metfacesu-1024x1024.pkl'
    },
    'afhqv2-t': {
        'url': '{}/stylegan3-t-afhqv2-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/afhqv2/stylegan3-t-afhqv2-512x512.pkl'
    },
    'afhqv2-r': {
        'url': '{}/stylegan3-r-afhqv2-1024x1024.pkl'.format(base_url), 
        'output_path': 'stylegan3/pretrained/afhqv2/stylegan3-r-afhqv2-512x512.pkl'
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
        raise Exception("No pretrained model named %s. Run stylegan3.get_pretrained_models() to get a list." % model_name) 


def run(latents, labels, translate=[0, 0], rotate=0, noise_mode='const', truncation=1.0):
    if hasattr(G.synthesis, 'input'):
        m = gen_images.make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    images = G(latents, labels, truncation_psi=truncation, noise_mode=noise_mode)
    images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    images = images.cpu().numpy()                        
    return images, latents


def random_sample(num_images, label, translate=[0, 0], rotate=0, noise_mode='const', truncation=1.0, seed=None):
    seed = seed if seed else np.random.randint(1e8)
    rnd = np.random.RandomState(int(seed))
    latents = torch.from_numpy(rnd.randn(num_images, G.z_dim)).to(device)
    labels = None #np.zeros((num_images, 7))
    return run(latents, labels, translate, rotate, noise_mode, truncation)


def interpolated_matrix_between(start, end, num_frames):
    linfit = interp1d([0, num_frames-1], np.vstack([start, end]), axis=0)
    interp_matrix = np.zeros((num_frames, start.shape[1]))
    for f in range(num_frames):
        interp_matrix[f, :] = linfit(f)
    return interp_matrix


def get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps=30, seed=None):
    global G
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    random_state = np.random.RandomState(seed if seed is not None else np.random.randint(1000))
    shape = [num_frames, np.prod([1, 1])] + [G.z_dim] # [frame, image, channel, component]
    latents = random_state.randn(*shape).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [smoothing_sec * mp4_fps] + [0] * 2, mode='wrap')
    latents /= np.sqrt(np.mean(np.square(latents)))
    latents = torch.tensor(latents).to(device)
    return latents


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
        e1, e2 = e, (e+1)%num_endpoints
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
    latents = latents / np.array(nss).reshape((512,1))
    return latents.T


def generate_interpolation_video(output_path, all_latents, labels, truncation=1, duration_sec=60.0, smoothing_sec=1.0, image_shrink=1, image_zoom=1, mp4_fps=30, mp4_codec='libx264', mp4_bitrate='16M', seed=None, minibatch_size=16):    
    global Gs, Gs_syn_kwargs

    num_frames = int(np.rint(duration_sec * mp4_fps))    
    all_latents = get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps, seed)
    all_labels = get_interpolated_labels(labels, num_frames)

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        the_latents = all_latents[frame_idx]
        labels = None #all_labels[frame_idx].reshape((1, 7))
        translate, rotate = [0, 0], 0
        if hasattr(G.synthesis, 'input'):
            m = gen_images.make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        images = G(the_latents, labels, truncation_psi=truncation, noise_mode=noise_mode)
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images[0].cpu().numpy()
        return images
        
    clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    clip.write_videofile(output_path, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    return output_path


def generate_interpolation_video2(output_path, labels, truncation=1, noise_mode='const', duration_sec=60.0, smoothing_sec=1.0, image_shrink=1, image_zoom=1, mp4_fps=30, mp4_codec='libx264', mp4_bitrate='16M', seed=None, minibatch_size=16):    
    global G

    num_frames = int(np.rint(duration_sec * mp4_fps))    
    all_latents = get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps, seed)
    all_labels = get_interpolated_labels(labels, num_frames)
    
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        the_latents = all_latents[frame_idx]
        labels = None #all_labels[frame_idx].reshape((1, 7))
        translate, rotate = [0, 0], 0
        if hasattr(G.synthesis, 'input'):
            m = gen_images.make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        images = G(the_latents, labels, truncation_psi=truncation, noise_mode=noise_mode)
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images[0].cpu().numpy()
        return images
        
    clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    clip.write_videofile(output_path, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    return output_path

#    cmd = 'ffmpeg -y -i "%s" -c:v libx264 -pix_fmt yuv420p "%s";ls "%s"' % (os.path.join(result_subdir, mp4_name_temp), os.path.join(result_subdir, mp4_name), os.path.join(result_subdir, mp4_name_temp))
#    os.system(cmd)
    


def load_model(network_pkl):
    global G, device
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

