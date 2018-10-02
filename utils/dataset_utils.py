import os
from os import listdir
from os.path import isfile, join
from random import random, sample
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import invert
import cv2
from hed import *


#args
# randomize training/test (across augmentation also)
# rotation
# simplify tracing
# output_format rgb, g
# output_size [w, h] or keep the same
# crops/centers
#  - to crop or to hard-resize
# duplication + augmentation (sheer, rotate, etc)
# output save format (jpg png)



allowable_actions = ['none', 'quantize', 'trace', 'hed', 'segment', 'simplify']

parser = argparse.ArgumentParser()

# input, output
parser.add_argument("--input_dir", help="where to get input images")
parser.add_argument("--output_dir", help="where to put output images")
parser.add_argument("--num_images", type=int, help="number of images to take (omit to use all)", default=None)
parser.add_argument("--shuffle", action="store_true", help="shuffle image")


# processing action
parser.add_argument("--action", type=str, help="comma-separated: lis of actions from {%s} to take, e.g. trace,hed" % ','.join(allowable_actions), required=True, default="")

# augmentation
parser.add_argument("--augment", action="store_true", help="to augment or not augment")
parser.add_argument("--num_augment", type=int, help="number of regions to output", default=1)
parser.add_argument("--frac", type=float, help="cropping ratio before resizing", default=0.6667)
parser.add_argument("--frac_vary", type=float, help="cropping ratio vary", default=0.075)
parser.add_argument("--max_ang", type=float, help="max rotation angle (degrees)", default=0)
parser.add_argument("--max_stretch", type=float, help="maximum stretching factor (0=none)", default=0)
parser.add_argument("--w", type=int, help="output image width", default=64)
parser.add_argument("--h", type=int, help="output image height", default=64)
parser.add_argument("--min_dim", type=int, help="minimum width/height to allow for images", default=256)

# augmentation
parser.add_argument("--split", action="store_true", help="to split into training/test")
parser.add_argument("--pct_train", type=float, default=0.9, help="percentage that goes to training set")
parser.add_argument("--combine", action="store_true", help="concatenate input and output images (like for training pix2pix)")
parser.add_argument("--include_orig", action="store_true", help="if combine==0, include original?")

# etc
parser.add_argument("--hed_model_path", type=str, default='../data/HED_reproduced.npz', help="model path for HED (default ../data)")



def cv2pil(img):
    if len(img.shape) == 2 or img.shape[2]==1:
        cv2_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im.astype('uint8'))
    return pil_im

def pil2cv(img):
    pil_image = img.convert('RGB') 
    cv2_image = np.array(pil_image) 
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image
    
def try_make_dir(new_dir):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

## Operations

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
    w, h = img.width, img.height
    size_thresh = 0.001 * w * h
    img = pil2cv(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = run_hed(cv2pil(img), hed_model_path)
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


def crop_rot_resize(img, frac, w2, h2, ang, stretch):
    if img.height<h2 or img.width<w2:
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

    #xr, yr = 0.275 + 0.45*random(), 0.275 + 0.45*random()
    xr, yr = random(), random()
    x_crop, y_crop = (w1 - w1_crop - 1) * xr, (h1 - h1_crop - 1) * yr
    h1_crop, w1_crop, y_crop, x_crop = int(h1_crop), int(w1_crop), int(y_crop), int(x_crop)
    img_crop = img.crop((x_crop, y_crop, x_crop+w1_crop, y_crop+h1_crop))
    img_resize = img_crop.resize((w2, h2))
    
    return img_resize


def augmentation(img, args):
    num, w2, h2, frac, frac_vary, max_ang, max_stretch = args.num_augment, args.w, args.h, args.frac, args.frac_vary, args.max_ang, args.max_stretch
    aug_imgs = []
    for n in range(num):
        ang = max_ang * (-1.0 + 2.0 * random())
        frac_amt = frac + frac_vary * (-1.0 + 2.0 * random())
        stretch = max_stretch * (-1.0 + 2.0 * random())
        aug_img = crop_rot_resize(img, frac_amt, w2, h2, ang, stretch)
        aug_imgs.append(aug_img)
    return aug_imgs
    

    
    
# main program    
def main(args):
    action, num_images, min_w, min_h, pct_train, augment, split, combine, include_orig, shuffle, hed_model_path = args.action, args.num_images, args.min_dim, args.min_dim, args.pct_train, args.augment, args.split, args.combine, args.include_orig, args.shuffle, args.hed_model_path

    # get list of actions
    actions = action.split(',')
    if False in [a in allowable_actions for a in actions]:
        raise Exception('one of your actions does not exist')
    
    # I/O directoris
    input_dir, output_dir = args.input_dir, args.output_dir
    output_train_dir = join(output_dir, 'train')
    output_test_dir = join(output_dir, 'test')
    output_trainA_dir = join(output_dir,'train/train_A')
    output_trainB_dir = join(output_dir,'train/train_B')
    output_testA_dir = join(output_dir,'test/test_A')
    output_testB_dir = join(output_dir,'test/test_B')

    # which directories to make
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    if split:
        try_make_dir(output_train_dir)
        try_make_dir(output_test_dir)
        
    if include_orig and combine==False:
        try_make_dir(output_trainA_dir)
        try_make_dir(output_trainB_dir)
        if pct_train < 1.0:
            try_make_dir(output_testA_dir)
            try_make_dir(output_testB_dir)
    
    # cycle through input images
    images = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) ]
    sort_order = sorted(range(min(num_images if num_images is not None else 1e8, len(images))))
    if shuffle:
        sort_order = sorted(sample(range(len(images)), min(num_images if num_images is not None else 1e8, len(images))))
    images = [images[i] for i in sort_order]
    
    # if to split into training/test flders
    training = [1] * len(images)
    if split:
        n_train = int(len(images) * args.pct_train)
        training[n_train:] = [0] * (len(images) - n_train)
    
    for img_idx, img_path in enumerate(tqdm(images)):

        print('open %d/%d : %s' % (img_idx, len(images), join(input_dir, img_path)))
        ext = 'png' #img_path.split('.')[-1]
        img0 = Image.open(join(input_dir, img_path)).convert("RGB")

        if img0.width < min_w or img0.height < min_h:
            #print('skipping, too small (%d x %d)' % (img0.width, img0.height))
            continue

        imgs0 = []
        if augment:
            imgs0 = augmentation(img0, args)
        else:   
            imgs0 = [img0]

        imgs = []
        for img0 in tqdm(imgs0):            

            img = img0
            
            for a in actions:
                if a == 'segment':
                    img = segment(img)
                elif a == 'colorize':
                    colors = [[255,255,255], [0,0,0], [127,0,0], [0, 0, 127], [0, 127, 0]]
                    img = quantize_colors(img, colors)
                elif a == 'trace':
                    img = trace(img)
                elif a == 'hed':
                    img = run_hed(img, hed_model_path)
                elif a == 'simplify':
                    img = simplify(img, hed_model_path)
                elif a == 'none' or a == '':
                    pass

            imgs.append(img)

        for i, (img0, img1) in enumerate(zip(imgs0, imgs)):
            out_dir = join(output_dir, 'train' if training[img_idx]==1 else 'test') if split else output_dir
            out_dir_orig = join(output_dir, 'train/train_B' if training[img_idx]==1 else 'test/test_B') if split else output_dir
            if include_orig and combine==False:
                out_dir = join(output_dir, 'train/train_A' if training[img_idx]==1 else 'test/test_A') if split else output_dir
            out_img = '%05d_%s_%d.%s' % (img_idx, ''.join(img_path.split('.')[0:-1]), i, ext)
            if combine:                
                img_f = Image.new('RGB', (args.w * 2, args.h))     
                img_f.paste(img0.convert('RGB'), (0, 0))
                img_f.paste(img1.convert('RGB'), (args.w, 0))
                img_f.save(join(out_dir, out_img))
            elif include_orig:
                img1 = img1.convert('RGB')
                img1.save(join(out_dir, out_img))
                img0 = img0.convert('RGB')
                img0.save(join(out_dir_orig, out_img))
            else:
                img1 = img1.convert('RGB')
                img1.save(join(out_dir, out_img))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
