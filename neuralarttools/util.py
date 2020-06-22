from PIL import Image
import requests
import urllib.parse
from io import BytesIO
import IPython

    
    
class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def display_local_videos(videos):
    videos = videos if isinstance(videos, list) else [videos]
    html_str = ''
    for video in videos:
        html_str += '<video controls style="margin:2px" src="%s"></video>' % video
    IPython.core.display.display(IPython.core.display.HTML(html_str))    


def is_url(path):
    path = urllib.parse.urlparse(path)
    return path.netloc != ''


def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img






#########

from moviepy.editor import *


# def resize_frames_to_even(frames):
#     frames = [np.array(frame) for frame in frames]
#     w1, h1 = frames[0].shape[0:2]
#     if w1%2==1 or h1%2==1:
#         w2 = w1+1 if w1%2==1 else w1
#         h2 = h1+1 if h1%2==1 else h1
#         frames = [np.array(resize(frame, (w2, h2))) 
#                   for frame in frames]
#     return frames


def frames_to_movie(frames, fps=30):
    frames = [np.array(frame) for frame in frames]
    w1, h1 = frames[0].shape[0:2]
    if w1%2==1 or h1%2==1:
        w2 = w1+1 if w1%2==1 else w1
        h2 = h1+1 if h1%2==1 else h1
        frames = [np.array(resize(frame, (w2, h2))) 
                  for frame in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    disp_clip = clip.ipython_display()
    os.system('rm __temp__.mp4')
    IPython.display.clear_output()
    return disp_clip


#######
import numpy as np


def load_image(image, image_size=None):
    if isinstance(image, str):
        if is_url(image):
            image = url_to_image(image)
        elif os.path.exists(image):
            image = Image.open(image).convert('RGB')
        else:
            raise ValueError('no image found at %s'%image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    if image_size is not None and isinstance(image_size, tuple):
        image = resize(image, image_size)
    elif image_size is not None and not isinstance(image_size, tuple):
        aspect = get_aspect_ratio(image)
        image_size = (int(aspect * image_size), image_size)
        image = resize(image, image_size)
    return image


def resize(img, new_size, mode=None, align_corners=True):
    sampling_modes = {
        'nearest': Image.NEAREST, 
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC, 
        'lanczos': Image.LANCZOS
    }
    assert isinstance(new_size, tuple), \
        'Error: image_size must be a tuple.'
    assert mode is None or mode in sampling_modes.keys(), \
        'Error: resample mode %s not understood: options are nearest, bilinear, bicubic, lanczos.'
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    w1, h1 = img.size
    w2, h2 = new_size
    if (h1, w1) == (w2, h2):
        return img
    if mode is None:
        mode = 'bicubic' if w2*h2 >= w1*h1 else 'lanczos'
    resample_mode = sampling_modes[mode]
    return img.resize((w2, h2), resample=resample_mode)


def get_size(image):
    if isinstance(image, str):
        image = load_image(image, 1024)
        w, h = image.size
    elif isinstance(image, Image.Image):    
        w, h = image.size
    elif isinstance(image, np.ndarray):
        w, h = image.shape[1], image.shape[0]
    return w, h


def get_aspect_ratio(image):
    w, h = get_size(image)
    return float(w) / h


def display(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    IPython.display.display(img)

    
def save(img, filename):
    folder = os.path.dirname(filename)
    if folder and not os.path.isdir(folder):
        os.mkdir(folder)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    img.save(str(filename))

    
def save_frame(img, index, folder):
    filename = '%s/f%05d.png' % (folder, index)
    save(img, filename)

