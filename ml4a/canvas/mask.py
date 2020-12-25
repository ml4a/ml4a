import math
import os
import itertools
from tqdm import tqdm
import IPython
import numpy as np
import sklearn.cluster
import cv2

from ..models import basnet
from ..utils.downloads import *
from ..utils.console import *
from ..utils import EasyDict
from ..image import *
from .canvas import *


default_color_labels = [
    [255,0,0],     [0, 255, 0],   [0,0,255],
    [255,255,0],   [255,0,255],   [255,255,0],
    [0,0,0],       [255,255,255], [64,0,192],
    [192,64,0],    [0,192,64],    [192,0,64], 
    [64,192,0],    [0,64,192]
]


def mask_to_image(mask):
    mask = (255.0 * mask).astype(np.uint8)
    mask_pil = Image.fromarray(mask).convert('RGB')
    return mask_pil


def image_to_mask(img):    
    mask = np.array(img).astype(np.float32) / 255.0
    return mask


def generate_mask_frames(masks, flatten_blend=False, merge=True):
    masks = masks if isinstance(masks, list) else [masks]
    color_labels = default_color_labels
    frames = []
    for mask in masks:
        h, w, nc = mask.shape
        mask_arr = np.zeros((h, w * nc))
        for c in range(nc):
            mask_arr[:, c * w:(c+1)*w] = mask[:, :, c]
        if flatten_blend:
            mask_arr = 0.5*(mask_arr>0.0)+0.5*(mask_arr==1.0)
        if merge:    
            mask_sum = np.sum(mask, axis=2)
            mask_norm = mask / mask_sum[:, :, np.newaxis] 
            mask_frame = np.sum(
                [[mask_norm[:, :, c] * clr 
                  for clr in color_labels[c%len(color_labels)]] 
                 for c in range(nc)], 
                axis=0).transpose((1,2,0))
        else:
            mask_frame = 255 * mask if merge else 255 * mask_arr
        frames.append(mask_frame)
    return frames


def view_mask(masks, flatten_blend=False, merge=True, animate=True, fps=30):
    masks = masks if isinstance(masks, list) else [masks]
    masks = [get_mask(m, 0) if isinstance(m, dict) else m for m in masks]
    animate = animate if len(masks)>1 else False
    frames = generate_mask_frames(masks, flatten_blend, merge)
    if animate:
        return frames_to_movie(frames, fps=fps)
    else:
        for frame in frames:
            display(frame)
    
    
def save_mask_video(filename, masks, flatten_blend=False, merge=True, fps=30):
    frames = generate_mask_frames(masks, flatten_blend, merge)
    clip = ImageSequenceClip(frames, fps=fps)
    folder = os.path.dirname(filename)
    if folder and not os.path.isdir(folder):
        os.mkdir(folder)
    clip.write_videofile(filename, fps=fps)
    IPython.display.clear_output()
    

def mask_arcs(size, num_channels, center, radius, period, t, blend=0.0, inwards=False, reverse=False):
    blend += 1e-8  # hack to fix bugs
    (w, h), (ctr_x, ctr_y), n = size, center, num_channels    
    rad = radius * n
    mask = np.zeros((h, w, n))
    pts = np.array([[[i/(h-1.0),j/(w-1.0)] for j in range(w)] for i in range(h)])
    ctr = np.array([[[ctr_y, ctr_x] for j in range(w)] for i in range(h)])
    pts -= ctr
    dist = (pts[:,:,0]**2 + pts[:,:,1]**2)**0.5
    pct = (float(-t if inwards else period+t) / (n * period)) % 1.0
    d = (dist + rad * (1.0 - pct)) % rad
    for c in range(0, n):
        cidx = n-c-1 if (reverse != inwards) else c
        x1, x2 = radius * (n-c-1), radius * (n-c)
        x1b, x2b = x1 - d, d - x2
        dm = np.maximum(0, np.maximum(d-x2, x1-d)) 
        mask[:, :, cidx] = np.clip(1.0-x1b/(blend*radius), 0, 1) * np.clip(1.0-x2b/(blend*radius), 0, 1) if blend > 0 else (np.maximum(0, np.maximum(d-x2, x1-d)) <=0) * (dist < radius)
    return mask


def mask_rects(size, num_channels, p1, p2, width, period, t, blend=0.0, reverse=False):
    p2 = (p2[0] + 1e-8 if p2[0]==p1[0] else p2[0],
          p2[1] + 1e-8 if p2[1]==p1[1] else p2[1])  # hack to fix bugs
    blend += 1e-8  # hack to fix bugs
    (w, h), n = size, num_channels
    mask = np.zeros((h, w, n))
    length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
    m1 = 1e8 if p2[0]==p1[0] else (p2[1] - p1[1]) / (p2[0]-p1[0]) 
    b1 = p1[1] - m1 * p1[0]
    m2 = -1.0 / (m1+1e-8) #9e8 if m1==0 else (1e-8 if m1==9e8 else -1.0 / m1)
    pts = np.array([[[i/(h-1.0),j/(w-1.0)] for j in range(w)] for i in range(h)])
    x1, y1 = pts[:,:,0], pts[:,:,1]
    x_int = (y1 - m2 * x1 - b1) / (m1 - m2)
    y_int = m2 * x_int + y1 - m2 * x1
    isect = np.zeros(pts.shape)
    isect[:,:,0] = (y1 - m2 * x1 - b1) / (m1 - m2)
    isect[:,:,1] = m2 * isect[:,:,0] + y1 - m2 * x1
    inside = (isect[:,:,0] >= min(p1[0], p2[0])) * (isect[:,:,0] <= max(p1[0], p2[0])) 
    dist = ((isect[:,:,0]-p1[0])**2 + (isect[:,:,1]-p1[1])**2)**0.5 
    pts_from_isect = pts - isect
    dst_from_isect = ((pts_from_isect[:,:,0])**2 + (pts_from_isect[:,:,1])**2)**0.5 
    offset = length - length * float(t)/(n*period)
    dist = (dist + offset) % length
    dist_diag = (dist * inside) / length
    rad = 1.0 / n
    for r in range(n):
        ridx = n-r-1 if reverse else r
        t1, t2 = rad * (n-r-1), rad * (n-r)
        t1d, t2d = t1 - dist_diag, dist_diag - t2
        val = np.clip(1.0-t1d/(blend*rad), 0, 1) * np.clip(1.0-t2d/(blend*rad), 0, 1) if blend > 0 else (dist_diag >= t1)*(dist_diag<t2)
        val = val.astype(np.float32)
        dc = dst_from_isect - width/2.0
        val *= (dst_from_isect <= width/2.0)
        mask[:, :, ridx] = val
    return mask


def mask_identity(size, num_channels):
    w, h = size
    mask = np.ones((h, w, num_channels))
    return mask


def mask_interpolation(size, num_channels, period, t, blend=0.0, reverse=False, cross_fade=True):
    (w, h), n = size, num_channels
    mask = np.zeros((h, w, n))
    idx1 = int(math.floor(t / period) % n) if period > 0 else 0
    idx2 = int((idx1 + 1) % n)
    if reverse:
        idx1 = n-idx1-1
        idx2 = n-idx2-1
    pct = float(t % period) / period if period > 0 else 0
    progress = min(1.0, float(1.0 - pct) / blend) if blend > 0 else (1.0 - pct)
    t2 = 1.0 - progress # * progress
    t1 = 1.0 - t2 if cross_fade else 1.0
    mask[:, :, idx1] = t1
    mask[:, :, idx2] = t2
    return mask


def mask_image_manual(size, num_channels, image, thresholds, blur_k, n_dilations):
    (w, h), n = size, num_channels
    assert len(thresholds) == n, 'Number of thresholds doesn\'t match number of channels in mask'
    mask = np.zeros((h, w, n))
    img = load_image(image) if isinstance(image, str) else image
    img = resize(img, size)
    img = np.array(img)[:, :, ::-1]
    img = crop_to_aspect_ratio(img, float(w)/h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.blur(img, (blur_k, blur_k))
    cumulative = np.zeros(img.shape[:2]).astype('uint8')
    for channel, thresh in enumerate(thresholds):
        ret, img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
        img1 -= cumulative
        cumulative += img1
        for d in range(n_dilations):
            img1 = cv2.dilate(img1, (2, 2))    
        ih, iw = img.shape
        img1 = cv2.blur(img1, (blur_k, blur_k))
        img1 = cv2.resize(img1, (w, h))
        mask[:,:,channel] += img1/255.
    return mask


def mask_image_auto(size, num_channels, image, blur_k):
    (w, h), n = size, num_channels
    mask = np.zeros((h, w, n))
    img = load_image(image) if isinstance(image, str) else image
    img = resize(img, size)
    img = np.array(img)[:, :, ::-1]
    img = crop_to_aspect_ratio(img, float(w)/h)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.blur(img, (blur_k, blur_k))
    mask_cumulative = 255 * img.shape[0] * img.shape[1] / num_channels
    cumulative = np.zeros(img.shape[0:2]).astype(np.uint8)
    thresh, thresholds = 0, []
    for channel in range(n):
        amt_mask = 0
        while amt_mask < mask_cumulative and thresh<=256:
            thresh += 1
            ret, img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
            #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            #cumulative = cumulative.reshape((780,1035,1))
            img1 -= cumulative
            amt_mask = np.sum(img1)
            #print(channel, thresh, amt_mask, mask_cumulative)
        cumulative += img1
        img1 = cv2.blur(img1, (blur_k, blur_k))
        img1 = cv2.resize(img1, (w, h))
        thresholds.append(thresh)
        mask[:,:,channel] += img1/255.
    return mask


def mask_image_kmeans(size, num_channels, image, blur_k, n_dilations, prev_mask=None):
    (w, h), n = size, num_channels
    mask = np.zeros((h, w, n))
    img = load_image(image) if isinstance(image, str) else image
    img = resize(img, size)
    img = np.array(img)[:, :, ::-1]
    img = cv2.blur(img, (blur_k, blur_k))
    img = crop_to_aspect_ratio(img, float(w)/h)
    #img = cv2.resize(img, (w, h), cv2.INTER_NEAREST)   # CHANGE
    img = np.array(resize(img, (w, h)))
    pixels = np.array(list(img)).reshape(h * w, 3)
    clusters, assign, _ = sklearn.cluster.k_means(pixels, n, init='k-means++', random_state=3425)
    
    if prev_mask is not None:        
        prev_assign = np.array(list(np.argmax(prev_mask, 2))).reshape(h * w)
        assign_candidates, best_total = list(itertools.permutations(range(n))), -1
        for ac in assign_candidates:
            reassign = np.array([ac[a] for a in assign])
            total = np.sum(reassign == prev_assign)
            if total > best_total:
                best_total = total 
                best_assign = reassign
        assign = best_assign
    else:
        amts = [np.sum(assign==c) for c in range(n)]
        order = list(reversed(sorted(range(len(amts)), key=lambda k: amts[k])))
        reorder = [order.index(i) for i in range(n)]
        assign = np.array([reorder[a] for a in assign])

    for c in range(n):
        channel_mask = np.multiply(np.ones((h*w)), assign==c).reshape((h,w))
        for d in range(n_dilations):
            channel_mask = cv2.dilate(channel_mask, (3, 3))    
        mask[:,:,c] = channel_mask

    return mask


def mask_image_basnet(size, image):
    (w, h), n = size, 2
    mask = np.zeros((h, w, n))
    img = load_image(image) if isinstance(image, str) else image
    img = resize(img, size)
    img = np.array(img)[:, :, ::-1]
    img = crop_to_aspect_ratio(img, float(w)/h)
    mask[:,:,0] = basnet.get_foreground(img)[:,:,0]/255.0
    mask[:,:,1] = 1.0-mask[:,:,0]
    return mask


def get_mask(mask, size=None, t=0):
    m = EasyDict(mask)
    
    if size is None and m.type != 'image':
        size = m.size if 'size' in m else (512, 512)

    m.num_channels = m.num_channels if 'num_channels' in m else 1
    m.period = m.period if 'period' in m else 1e8
    m.normalize = m.normalize if 'normalize' in m else False
    
    if m.type == 'solid': 
        masks = mask_identity(
            size=size, 
            num_channels=m.num_channels
        )
   
    elif m.type == 'interpolation':
        m.blend = m.blend if 'blend' in m else 0.0
        m.reverse = m.reverse if 'reverse' in m else False
        m.cross_fade = m.cross_fade if 'cross_fade' in m else False
        
        masks = mask_interpolation(
            size=size, 
            num_channels=m.num_channels, 
            period=m.period, t=t, 
            blend=m.blend, 
            reverse=m.reverse, 
            cross_fade=m.cross_fade
        )

    elif m.type == 'arcs':
        m.center = m.center if 'center' in m else (0.5, 0.5)
        m.radius = m.radius if 'radius' in m else 0.70710678118
        m.blend = m.blend if 'blend' in m else 0.0
        m.inwards = m.inwards if 'inwards' in m else False
        m.reverse = m.reverse if 'reverse' in m else False

        masks = mask_arcs(
            size=size, 
            num_channels=m.num_channels, 
            center=m.center, 
            radius=m.radius, 
            period=m.period, t=t, 
            blend=m.blend, 
            inwards=m.inwards, 
            reverse=m.reverse
        )

    elif m.type == 'rects':
        m.p1 = m.p1 if 'p1' in m else (0.0, 0.0)
        m.p2 = m.p2 if 'p2' in m else (1.0, 1.0)
        m.width = m.width if 'width' in m else 1.41421356237
        m.blend = m.blend if 'blend' in m else 0.0
        m.reverse = m.reverse if 'reverse' in m else False

        masks = mask_rects(
            size=size, 
            num_channels=m.num_channels, 
            p1=m.p1, p2=m.p2, 
            width=m.width, 
            period=m.period, t=t, 
            blend=m.blend, 
            reverse=m.reverse
        )

    elif m.type == 'image':
        m.blur_k = m.blur_k if 'blur_k' in m else 1
        m.n_dilations = m.n_dilations if 'n_dilations' in m else 0
        m.prev_mask = m.prev_mask if 'prev_mask' in m else None
        m.method = m.method if 'method' in m else 'kmeans'
        m.image = m.image if 'image' in m else '../neural-style-pt/images/inputs/monalisa.jpg'
        is_movie = isinstance(m.image, MoviePlayer) or type(m.image).__name__ == 'MoviePlayer'
        m.image = m.image.get_frame(t) if is_movie else m.image
        assert m.method in ['kmeans', 'threshold', 'auto', 'basnet'], \
            'Invalid method %s. Options are (kmeans, threshold, auto)'%m.method

        if size is None:
            size = get_size(m.image)
        
        if m.method == 'kmeans':        
            masks = mask_image_kmeans(
                size=size, 
                num_channels=m.num_channels, 
                image=m.image, 
                blur_k=m.blur_k,
                n_dilations=m.n_dilations,
                prev_mask=m.prev_mask
            )

        elif m.method == 'threshold':
            masks = mask_image_manual(
                size=size, 
                num_channels=m.num_channels, 
                image=m.image, 
                thresholds=m.thresholds, 
                blur_k=m.blur_k, 
                n_dilations=m.n_dilations
            )
            
        elif m.method == 'auto':
            masks = mask_image_auto(
                size=size, 
                num_channels=m.num_channels, 
                image=m.image, 
                blur_k=m.blur_k
            )

        elif m.method == 'basnet':
            masks = mask_image_basnet(
                size=size, 
                image=m.image
            )

    if m.normalize:
        mask_sum = np.maximum(
            np.ones(masks.shape[0:2]), 
            np.sum(masks, axis=2)
        )
        masks = masks / mask_sum[:, :, np.newaxis] 

    return masks


def mask_image(img, mask):
    img, mask = np.array(img), np.array(mask)/255.0
    img = img.reshape(*img.shape, 1) if img.ndim == 2 else img
    mask = mask.reshape(*mask.shape, 1) if mask.ndim == 2 else mask
    nc_img, nc_mask = img.shape[-1], mask.shape[-1]
    if nc_mask == 1:
        mask = mask[:, :] * np.ones(nc_img, dtype=int)[None, None, :]
    nc_img, nc_mask = img.shape[-1], mask.shape[-1]
    return img * mask
