import os
import random
import math
import PIL.Image
import moviepy.editor
import matplotlib.pyplot as pyplot
import scipy
from scipy.interpolate import interp1d
import numpy as np

from ..utils import downloads
from .. import image
from . import submodules

# todo: absorb latent code to ml4a.utils.latents

cuda_available = submodules.cuda_available()

#with submodules.localimport('submodules/stylegan2') as _importer:
with submodules.import_from('stylegan2'):  # localimport fails here
    import pretrained_networks
    import dnnlib
    import dnnlib.tflib as tflib

_G, _D, Gs, Gs_syn_kwargs = None, None, None, None

pretrained_models = {
    'cars': {
        'gdrive_fileid': '1jmEmT_LFbVnxk-jBU52KRNq8n2jBRbUk', 
        'output_path': 'stylegan2/pretrained/cars/stylegan2-car-config-f.pkl'
    },
    'cats': {
        'gdrive_fileid': '1Di_Kgs09KMtFWrjiUmcCx9kNbDNxgRUj', 
        'output_path': 'stylegan2/pretrained/cats/stylegan2-cat-config-f.pkl'
    },
    'churches': {
        'gdrive_fileid': '1Rm0tYPxfRvGA4ZPe5QbDu_iCJv9PVm5h', 
        'output_path': 'stylegan2/pretrained/churches/stylegan2-church-config-f.pkl'
    },
    'horses': {
        'gdrive_fileid': '1DQfp4YjeMvbSSBwc9qKO9D8FLRBsKBZD', 
        'output_path': 'stylegan2/pretrained/horses/stylegan2-horse-config-f.pkl'
    },
    'ffhq': {
        'gdrive_fileid': '1qSJvpFOUf6vSjAzmjjRya21Hlc9dCC9S', 
        'output_path': 'stylegan2/pretrained/ffhq/network-final-ffhq.pth'
    },    
    'landscapes': {
        'gdrive_fileid': '1UV6dUphjG8kUyYS4FWpzwTHpuGcdQ62Y', 
        'output_path': 'stylegan2/pretrained/landscapes/network-final-landscapes.pth'
    },
    'wikiarts': {
        'gdrive_fileid': '1Kg7yqWSgoXN_mvHypX_fXt2GjrJZ_CZv', 
        'output_path': 'stylegan2/pretrained/wikiarts/network-final-wikiarts.pth'
    }
}


def get_pretrained_models():
    return pretrained_models.keys()


def get_pretrained_model(model_name):
    if model_name in pretrained_models:
        model = pretrained_models[model_name]
        model_file = downloads.download_from_gdrive(
            gdrive_fileid=model['gdrive_fileid'], 
            output_path=model['output_path'])
        return model_file
    else:
        raise Exception("No pretrained model named %s. Run stylegan2.get_pretrained_models() to get a list." % model_name) 
    

# def display(images, num_cols=4,title=None):
#     n = len(images)
#     h, w, _ = images[0].shape
#     nr, nc = math.ceil(n / num_cols), num_cols
#     for r in range(nr):
#         idx1, idx2 = num_cols * r, min(n, num_cols * (r + 1))
#         img1 = np.concatenate([img for img in images[idx1:idx2]], axis=1)
#         if title is not None:
#             pyplot.title(title)        
#         pyplot.figure(figsize=(int(4 * float(w)/h * num_cols), 4))
#         pyplot.imshow(img1)

       
def run(latents, labels, truncation=1.0):
    images = Gs.run(latents, labels, truncation_psi=truncation, minibatch_size=8, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    return images, latents


def random_sample(num_images, label, truncation=1.0, seed=None):
    seed = seed if seed else np.random.randint(100)
    rnd = np.random.RandomState(int(seed))
    latents = rnd.randn(num_images, *Gs.input_shape[1:]) # [minibatch, component]
    labels = np.zeros((num_images, 7))
    return run(latents, labels, truncation)


def interpolated_matrix_between(start, end, num_frames):
    linfit = interp1d([0, num_frames-1], np.vstack([start, end]), axis=0)
    interp_matrix = np.zeros((num_frames, start.shape[1]))
    for f in range(num_frames):
        interp_matrix[f, :] = linfit(f)
    return interp_matrix


def get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps=30, seed=None):
    global Gs
    num_frames = int(np.rint(duration_sec * mp4_fps))    
    random_state = np.random.RandomState(seed if seed is not None else np.random.randint(1000))
    shape = [num_frames, np.prod([1, 1])] + Gs.input_shape[1:] # [frame, image, channel, component]
    latents = random_state.randn(*shape).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    latents /= np.sqrt(np.mean(np.square(latents)))
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
        labels = all_labels[frame_idx].reshape((1, 7))
        images = Gs.run(the_latents, labels, truncation_psi=truncation, minibatch_size=minibatch_size, **Gs_syn_kwargs)
        return images[0]
        
    clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    clip.write_videofile(output_path, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    return output_path

def generate_interpolation_video2(output_path, labels, truncation=1, duration_sec=60.0, smoothing_sec=1.0, image_shrink=1, image_zoom=1, mp4_fps=30, mp4_codec='libx264', mp4_bitrate='16M', seed=None, minibatch_size=16):    
    global Gs, Gs_syn_kwargs

    num_frames = int(np.rint(duration_sec * mp4_fps))    
    all_latents = get_gaussian_latents(duration_sec, smoothing_sec, mp4_fps, seed)
    all_labels = get_interpolated_labels(labels, num_frames)

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        the_latents = all_latents[frame_idx]
        labels = all_labels[frame_idx].reshape((1, 7))
        images = Gs.run(the_latents, labels, truncation_psi=truncation, minibatch_size=minibatch_size, **Gs_syn_kwargs)
        return images[0]
        
    clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    clip.write_videofile(output_path, fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    return output_path

#    cmd = 'ffmpeg -y -i "%s" -c:v libx264 -pix_fmt yuv420p "%s";ls "%s"' % (os.path.join(result_subdir, mp4_name_temp), os.path.join(result_subdir, mp4_name), os.path.join(result_subdir, mp4_name_temp))
#    os.system(cmd)
    


def load_model(network_pkl, randomize_noise=False):
    global _G, _D, Gs, Gs_syn_kwargs
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = randomize_noise

