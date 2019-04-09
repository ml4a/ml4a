import os
from random import random, sample
import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import felzenszwalb
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import invert
from tqdm import tqdm
import cv2


def cv2pil(cv2_img):
    if len(cv2_img.shape) == 2 or cv2_img.shape[2]==1:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img.astype('uint8'))
    return pil_img


def pil2cv(pil_img):
    pil_img = pil_img.convert('RGB') 
    cv2_img = np.array(pil_img) 
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    cv2_img = cv2_img[:, :, ::-1].copy()
    return cv2_img
    
    
def posterize(im, n):
    indices = np.arange(0,256)   # List of all colors 
    divider = np.linspace(0,255,n+1)[1] # we get a divider
    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
    palette = quantiz[color_levels] # Creating the palette
    im2 = palette[im]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2) # Converting image back to uint8
    return im2


def canny(im1):
    im1 = pil2cv(im1)
    im2 = cv2.GaussianBlur(im1, (5, 5), 0)
    im2 = cv2.Canny(im2, 100, 150)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    im2 = cv2pil(im2)
    return im2


def image2colorlabels(img, colors):
    h, w = img.height, img.width
    pixels = np.array(list(img.getdata()))
    dists = np.array([np.sum(np.abs(pixels-c), axis=1) for c in colors])
    classes = np.argmin(dists, axis=0)
    
    
def colorize_labels(img, colors):
    h, w = img.height, img.width
    classes = image2colorlabels(img, colors)
    img = Image.fromarray(np.uint8(classes.reshape((h, w, 3))))
    return img


def quantize_colors(img, colors):
    h, w = img.height, img.width
    classes = image2colorlabels(img, colors)
    pixels_clr = np.array([colors[p] for p in classes]).reshape((h, w, 3))
    img = Image.fromarray(np.uint8(pixels_clr))
    return img


def segment(img):
    img = pil2cv(img)
    h, w = img.shape[0:2]
    img = cv2.bilateralFilter(img, 9, 100, 100)
    scale = int(h * w / 1000)
    segments = felzenszwalb(img, scale=scale, sigma=0.5, min_size=150)
    out_image = np.zeros((h, w, 3))
    num_segments = len(np.unique(segments))
    for s in tqdm(range(num_segments)):
        label_map = segments==s
        label_map3 = np.dstack([label_map] * 3)
        masked_img = np.multiply(label_map3, img)
        #avg_color = np.sum(np.sum(masked_img, axis=0), axis=0) / np.count_nonzero(label_map)  # maybe median is better
        nonzeros = [ masked_img[:, :, c].reshape((h * w)) for c in range(3) ]
        median_color = [ np.median(np.take(nonzeros[c], nonzeros[c].nonzero())) for c in range(3) ]
        smooth_segment = (label_map3 * median_color).astype('uint8')
        out_image += smooth_segment
    out_image = Image.fromarray(out_image.astype('uint8'))
    return out_image


def trace(img):
    img = pil2cv(img)
    im2 = cv2.GaussianBlur(img, (5, 5), 0)
    im3 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    ret, im4 = cv2.threshold(im3, 127, 255, 0)
    ret, img = cv2.threshold(im3, 255, 255, 0)
    im5, contours, hierarchy = cv2.findContours(im4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [ c for c in contours if cv2.arcLength(c, True) > 8 ] #and cv2.contourArea(c) > 10]
    for contour in contours:
        cv2.drawContours(img, [contour], 0, (255), 2)
    img = cv2pil(img)
    return img


def simplify(img, hed_model_path):
    import hed_processing
    w, h = img.width, img.height
    size_thresh = 0.001 * w * h
    img = pil2cv(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = hed_processing.run_hed(cv2pil(img), hed_model_path)
    ret, img = cv2.threshold(pil2cv(img), 50, 255, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = remove_small_objects(img.astype('bool'), size_thresh)
    img = 255 * skeletonize(img).astype('uint8')
    img = cv2pil(img)
    return img
    
    
def upsample(img, w2, h2):
    h1, w1 = img.height, img.width
    r = max(float(w2)/w1, float(h2)/h1)
    img = img.resize((int(r*w1), int(r*h1)), resample=Image.BICUBIC)
    return img


    
def crop_rot_resize(img, frac, w2, h2, ang, stretch, centered):
    if w2 is None:
        w2 = img.width
    if h2 is None:
        h2 = img.height
    
    if img.height < h2 or img.width < w2:
        img = upsample(img, w2, h2)
    
    if stretch != 0:
        v = random() < 0.5
        h = 1.0 if not v else (1.0 + stretch)
        w = 1.0 if v else (1.0 + stretch)
        img = img.resize((int(img.width * w), int(img.height * h)), resample=Image.BICUBIC)
        
    if ang > 0:
        img = img.rotate(ang, resample=Image.BICUBIC, expand=False)
   
    ar = float(w2 / h2)
    h1, w1 = img.height, img.width

    if float(w1) / h1 > ar:
        h1_crop = max(h2, h1 * frac)
        w1_crop = h1_crop * ar
    else:
        w1_crop = max(w2, w1 * frac)
        h1_crop = w1_crop / ar

    xr, yr = (0.5, 0.5) if centered else (random(), random())
    x_crop, y_crop = (w1 - w1_crop - 1) * xr, (h1 - h1_crop - 1) * yr
    h1_crop, w1_crop, y_crop, x_crop = int(h1_crop), int(w1_crop), int(y_crop), int(x_crop)
    img_crop = img.crop((x_crop, y_crop, x_crop+w1_crop, y_crop+h1_crop))
    img_resize = img_crop.resize((w2, h2), resample=Image.BICUBIC)
    
    return img_resize




