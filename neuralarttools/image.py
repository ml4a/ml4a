import os
from PIL import Image
import numpy as np
import requests
import urllib.parse
from io import BytesIO
import IPython
import moviepy.editor as mpe

Image.MAX_IMAGE_PIXELS = 1e9
    
    
class MoviePlayer:
    
    def __init__(self, path):
        self.path = path
        self.video = mpe.VideoFileClip(path)
        self.fps = self.video.fps
        self.duration = self.video.duration
        self.num_frames = int(self.fps * self.duration)
        
    def get_frame(self, frame_idx):
        time = frame_idx / self.fps
        img = self.video.get_frame(time)
        img = Image.fromarray(img)
        return img
    

    
def load_image(image, image_size=None, to_numpy=False, normalize=False):
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
    if to_numpy:
        image = np.array(image)
        if normalize:
            image = image / 255.0        
    return image


def random_image(image_size, margin=1.0, bias=128.0):
    w, h = image_size
    img = bias + margin * (-1.0 + 2.0 * np.random.uniform(size=(h, w, 3)))
    return img


def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
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


def display(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    IPython.display.display(img)


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

    
