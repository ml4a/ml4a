import os
import time
import math
from PIL import Image
import numpy as np
import requests
import urllib.parse
from io import BytesIO
import IPython
import moviepy.editor as mpe

from .utils import downloads

Image.MAX_IMAGE_PIXELS = 1e9

sample_images = {
    'escher_sphere.jpg': '1pXYUSTfVDhJSOES0L0LEhf8A1lB28S4t',
    'frida_kahlo.jpg': '1JbTCTqApHJW6m4Fw6cH_dtgcIp0oKXcB',
    'hokusai.jpg': '1j7UzCT7q3EKCkyKh5QgIApdMi3j_iAdT',
    'monalisa.jpg': '1WZgqATI5dKDwOpkLfOhuBOvn4zt4UcTa',
    'starry_night.jpg': '1LLV7lyuCPOCDa5c7CkWjSfeckXBou1W7',
    'teddybear_frame1.png': '1e9cPyDMdsVIF26RI6htlALExiHbCPiXW',
    'teddybear_frame2.png': '1iL8r4LxRKNZH9xf0qa1bLW-hDXU2-gT8',
    'the_scream.jpg': '1jrhDwRidBbgv7Ki2yBfyMhubbnD2LHAY',
    'tubingen.jpg': '19lf288HWdzgSP4B8OQfHPv1un9ljZVJR'
}


def load_image(img, image_size=None, to_numpy=False, normalize=False, autocrop=False):
    if isinstance(img, str):
        if is_url(img):
            img = url_to_image(img)
            if img is None:
                print("Error: no image returned")
                return None
        elif os.path.exists(img):
            img = Image.open(img).convert('RGB')
        else:
            raise ValueError('no image found at %s'%img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    if image_size is not None and isinstance(image_size, tuple):
        if autocrop:
            aspect = float(image_size[0])/image_size[1]
            img = crop_to_aspect_ratio(img, aspect)
        img = resize(img, image_size)
    elif image_size is not None and not isinstance(image_size, tuple):
        aspect = get_aspect_ratio(img)
        image_size = (int(aspect * image_size), image_size)
        img = resize(img, image_size)
    if to_numpy:
        img = np.array(img)
        if normalize:
            img = img / 255.0        
    return img


def random_image(image_size, margin=1.0, bias=128.0):
    w, h = image_size
    img = bias + margin * (-1.0 + 2.0 * np.random.uniform(size=(h, w, 3)))
    return img


def url_to_image(url):
    finished = False
    max_tries, n_tries = 5, 0
    img = None
    while not finished:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            finished = True
        except:
            time.sleep(5)
            n_tries += 1
            finished = n_tries >= 10
    return img


def is_url(path):
    path = urllib.parse.urlparse(path)
    return path.netloc != ''


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


def get_size(img):
    if isinstance(img, str):
        image = load_image(image, 1024)
        w, h = img.size
    elif isinstance(img, Image.Image):    
        w, h = img.size
    elif isinstance(img, np.ndarray):
        w, h = img.shape[1], img.shape[0]
    return w, h


def get_aspect_ratio(img):
    w, h = get_size(img)
    return float(w) / h


def crop_to_aspect_ratio(img, aspect_ratio):
    iw, ih = get_size(img)
    ar_img = get_aspect_ratio(img)
    if ar_img > aspect_ratio:
        iw2 = ih * aspect_ratio
        ix = (iw-iw2)/2
        img = np.array(img)[:,int(ix):int(ix+iw2)]
    elif ar_img < aspect_ratio:
        ih2 = float(iw) / aspect_ratio
        iy = (ih-ih2)/2
        img = np.array(img)[int(iy):int(iy+ih2),:]
    return img


def display(images, animate=False, title=None, num_cols=4):
    if isinstance(images, list):
        num_image_sizes = len(set([np.array(img).shape for img in images]))
        if num_image_sizes > 1:
            images = concatenate_images(images)        
        else:
            images = [np.array(img) for img in images]
        images = np.array(images)
    else:
        images = np.array(images)
    ndim = np.array(images).ndim
    multiple_images = False
    if ndim == 2:
        images = np.expand_dims(images, axis=-1)
        num_channels = 1
    elif ndim == 3:
        if images.shape[-1] in [1,3,4]:
            num_channels = images.shape[-1]
        else:
            multiple_images = True
            num_channels = 1
    elif ndim == 4:
        multiple_images = True
        num_channels = images.shape[-1]
    if multiple_images:
        images = [np.array(img) for img in images]
    else:
        images = np.expand_dims(images, axis=0)
    if animate:
        return frames_to_movie(images, fps=30)
    n = len(images)
    num_cols = min(n, num_cols)
    h, w = np.array(images[0]).shape[:2]
    nr, nc = math.ceil(n / num_cols), num_cols
    for r in range(nr):
        idx1, idx2 = num_cols * r, min(n, num_cols * (r + 1))
        img_row = np.concatenate([img for img in images[idx1:idx2]], axis=1)
        if num_channels == 1:
            img_row = np.repeat(img_row, 3, axis=-1)
            num_channels = 3
        whitespace = np.zeros((h, (num_cols-(idx2-idx1))*w, num_channels))
        img_row = np.concatenate([img_row, whitespace], axis=1)
        img_row = Image.fromarray(img_row.astype(np.uint8)).convert('RGB')
        if title is not None:
            print(title)
        IPython.display.display(img_row)
    

def display_local(files):
    files = files if isinstance(files, list) else [files]
    html_str = ''
    for filename in files:
        ext = os.path.splitext(filename)[-1].lower()
        if ext in ['.mp4', 'mov', '']:
            html_str += '<video controls style="margin:2px" src="%s"></video>' % filename
        elif ext in ['.png', '.jpg']:
            html_str += '<img style="margin:2px" src="%s"></img>' % filename
    IPython.core.display.display(IPython.core.display.HTML(html_str))    


def concatenate_images(images, margin=0, vertical=False):
    w, h = get_size(images[0])
    W = w+2*margin if vertical else margin+len(images)*(w+margin)
    H = h+2*margin if not vertical else margin+len(images)*(h+margin)
    images = [resize(image, (w, h)) for image in images]
    images_all = Image.new('RGB', (W, H))
    for i, image in enumerate(images):
        x, y = (margin, margin+(h+margin)*i) if vertical else (margin+(w+margin)*i, margin)
        images_all.paste(image, (x, y))
    return images_all


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


def frames_to_movie(frames, fps=30):
    frames = [np.array(frame) for frame in frames]
    w1, h1 = frames[0].shape[0:2]
    if w1%2==1 or h1%2==1:
        w2 = w1+1 if w1%2==1 else w1
        h2 = h1+1 if h1%2==1 else h1
        frames = [np.array(resize(frame, (w2, h2))) 
                  for frame in frames]
    clip = mpe.ImageSequenceClip(frames, fps=fps)
    disp_clip = clip.ipython_display()
    os.system('rm __temp__.mp4')
    IPython.display.clear_output()
    return disp_clip


def load_sample_image(filename, size=None):
    assert filename in get_sample_images(), \
        '%s not found in sample images. Available images are %s' % (filename, ', '.join(get_sample_images()))
    sample_image_path = downloads.download_from_gdrive(
        gdrive_fileid=sample_images[filename],
        output_path='_data/sample_images/%s'%filename)
    return load_image(sample_image_path, size)


def get_sample_images():
    return sample_images.keys()


def escher(size=None): return load_sample_image('escher_sphere.jpg', size)
def fridakahlo(size=None): return load_sample_image('frida_kahlo.jpg', size)
def hokusai(size=None): return load_sample_image('hokusai.jpg', size)
def monalisa(size=None): return load_sample_image('monalisa.jpg', size)
def starrynight(size=None): return load_sample_image('starry_night.jpg', size)
def scream(size=None): return load_sample_image('the_scream.jpg', size)
def tubingen(size=None): return load_sample_image('tubingen.jpg', size)
        

class MoviePlayer:
    
    def __init__(self, path, t1=None, t2=None):
        self.path = path
        self.video = mpe.VideoFileClip(path)
        self.fps = self.video.fps
        self.duration = self.video.duration
        self.num_frames = int(self.fps * self.duration)
        self.frame1 = self.fps * t1 if t1 is not None else 0
        self.frame2 = self.fps * t2 if t2 is not None else self.num_frames
        
    def get_frame(self, frame_idx, size=None):
        frame_idx = self.frame1 + (frame_idx % (self.frame2-self.frame1))
        time = frame_idx / self.fps
        img = self.video.get_frame(time)
        img = Image.fromarray(img)
        if size is not None:
            img = resize(img)
        return img
    

