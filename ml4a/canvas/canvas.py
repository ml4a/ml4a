import math
import os
from tqdm import tqdm
import numpy as np
from io import BytesIO
from PIL import Image
import IPython
from moviepy.editor import *
from noise import pnoise2, snoise2

from ..utils.downloads import *
from ..utils.console import *
from ..image import *
from .mask import *


def get_canvas_frames(canvas, size, numframes, img=None):
    w, h = size
    if img is None and type(img) == str:
        img = load_image(img, (w, h))
    elif img is None:
        img = make_image_grid((w, h))
    else:
        img = resize(img, (w, h))
    frames = []
    for f in range(numframes):
        frames.append(np.array(img).astype(np.uint8))
        img = modify_canvas(img, canvas)
    return frames
    

def view_canvas(canvas, size, numframes, img=None, animate=False, fps=30):
    frames = get_canvas_frames(canvas, size, numframes, img)
    if animate:
        return frames_to_movie(frames, fps=fps)
    else: 
        for frame in frames:
            display(frame)
        return None

    
def save_canvas_video(filename, canvas, size, numframes, img, fps=30):
    frames = get_canvas_frames(canvas, size, numframes, img)
    clip = ImageSequenceClip(frames, fps=fps)
    folder = os.path.dirname(filename)
    if folder and not os.path.isdir(folder):
        os.mkdir(folder)
    clip.write_videofile(filename, fps=fps)
    IPython.display.clear_output()

    
def make_image_grid(size, spacing=15, thickness=3):
    w, h = size
    img = np.zeros((h, w, 3))
    for off in range(thickness):
        img[:,range(off, w, spacing),0] = 255
        img[range(off, h, spacing),:,0] = 255
    img[range(-8+int(h/2),8+int(h/2)),range(-8+int(w/2),8+int(w/2)),:] = [0,255,0]
    img[range(8+int(h/2),-8+int(h/2),-1),range(-8+int(w/2),8+int(w/2)),:] = [0,255,0]
    return img


def lerp_mod(mod1, mod2, r):
    mod_avg = {}
    for k in mod1:
        try:
            mod_avg[k] = (1.0 - r) * mod1[k] + r * mod2[k]
        except:
            mod_avg[k] = tuple([ (1.0 - r) * m1_ + r * m2_ for m1_, m2_ in zip(mod1[k], mod2[k]) ])
    return mod_avg


# def warp_image(mod, img_in, img_blend=None, amt_blend=None):
#     img_out = modify_canvas(img_in, mod)
#     if (img_blend is not None and amt_blend is not None and amt_blend > 0):
#         img_out = (1.0 - amt_blend) * img_out + amt_blend * img_blend
#     return img_out

def blend_images(img1, img2, blend_amt):
    img_out = (1.0 - blend_amt) * img1 + blend_amt * img2
    return img_out


def map_image(img, idx):  # should it be mod by h-1, w-1?
    img = np.array(img)
    h, w = img.shape[0:2]
    idx = np.array(idx).reshape((h*w, 2))
    idx_mod = np.mod(idx,[h,w])  
    idx_turn = np.mod(np.floor_divide(idx,[h,w]),2)
    idx = np.multiply(idx_mod,1-idx_turn)+np.multiply([h,w]-idx_mod,idx_turn)
    idx = np.clip(idx, [0,0], [h-1,w-1])
    idx = np.array(idx).astype(np.float32)
    idx_tl = np.floor(np.copy(idx).astype(np.float32)).astype(np.int32)
    idx_br = np.ceil(np.copy(idx).astype(np.float32)).astype(np.int32)
    idx_tr = np.copy(idx_tl).astype(np.int32)
    idx_bl = np.copy(idx_br).astype(np.int32)
    idx_tr[:,1] = idx_br[:,1]
    idx_bl[:,1] = idx_tl[:,1]
    diff = np.subtract(idx, idx_tl)
    dy, dx = diff[:, 0].reshape((h*w, 1)), diff[:, 1].reshape((h*w, 1))
    img_tl = img[idx_tl[:,0], idx_tl[:,1], :]
    img_tr = img[idx_tr[:,0], idx_tr[:,1], :]
    img_bl = img[idx_bl[:,0], idx_bl[:,1], :]
    img_br = img[idx_br[:,0], idx_br[:,1], :]
    img_t = np.add(np.multiply(np.subtract(1.0, dx), img_tl), np.multiply(dx, img_tr))
    img_b = np.add(np.multiply(np.subtract(1.0, dx), img_bl), np.multiply(dx, img_br))
    img_f = np.add(np.multiply(np.subtract(1.0, dy), img_t), np.multiply(dy, img_b))
    img_f = img_f.reshape((h, w, 3))
    return img_f


def modify_canvas(img, mods, masks=None, to_pil=True):
    img = np.array(img)
    h, w = img.shape[0:2]
    mods = mods if isinstance(mods, list) else [mods]
    masks = masks / np.sum(masks, axis=2)[:, :, np.newaxis] if masks is not None else np.ones((h, w, len(mods)))

    # make default grid
    grid = np.mgrid[0:w, 0:h].T
    grid = grid.astype(np.float32)    
    grid[:,:,0], grid[:,:,1] = grid[:,:,1], grid[:,:,0].copy()
    mod_idxs = np.copy([[grid]*3]*len(mods))
    
    # calculate all th index transformations
    for idxm, mod in enumerate(mods):

        mod['center'] = mod['center'] if 'center' in mod else (0.5, 0.5)
        mod['shift'] = mod['shift'] if 'shift' in mod else (0.0, 0.0)
        mod['stretch'] = mod['stretch'] if 'stretch' in mod else (1.0, 1.0)
        mod['zoom'] = mod['zoom'] if 'zoom' in mod else 1.0
        mod['expand'] = mod['expand'] if 'expand' in mod else 0.0
        mod['rot_const'] = mod['rot_const'] if 'rot_const' in mod else 0.0
        mod['rot_ang'] = mod['rot_ang'] if 'rot_ang' in mod else 0.0
        mod['rot_dst'] = mod['rot_dst'] if 'rot_dst' in mod else 0.0
        mod['spiral_margin'] = mod['spiral_margin'] if 'spiral_margin' in mod else 0.0
        mod['spiral_periods'] = mod['spiral_periods'] if 'spiral_periods' in mod else 0
        mod['noise_rate'] = mod['noise_rate'] if 'noise_rate' in mod else (0.0, 0.0)
        mod['noise_margin'] = mod['noise_margin'] if 'noise_margin' in mod else (0.0, 0.0)
        mod['noise_offset'] = mod['noise_offset'] if 'noise_offset' in mod else (-10.0, 10.0)
        
        shift, stretch = mod['shift'], mod['stretch']
        zoom, expand = mod['zoom'], mod['expand']
        rot_const, rot_ang, rot_dst = mod['rot_const'], mod['rot_ang'], mod['rot_dst']
        spiral_margin, spiral_periods = mod['spiral_margin'], mod['spiral_periods']
        noise_rate, noise_margin, noise_offset = mod['noise_rate'], mod['noise_margin'], mod['noise_offset']
        cy, cx = h * mod['center'][0], w * mod['center'][1]
        
        # check in advance on what operations so as to save time
        to_shift = (shift[0] != 0.0 or shift[1] != 0.0 or stretch[0] != 1.0 or stretch[1] != 1.0)
        to_zoom = (zoom != 1.0 or expand != 0.0)
        to_spiral = (spiral_margin > 0)
        to_rotate = (rot_const != 0.0 or rot_ang != 0.0 or rot_dst != 0.0)
        to_noise = (noise_margin[0] != 0.0 or noise_margin[0] != 0.0)

        # create initial grid for transformations
        if to_shift or to_zoom or to_spiral or to_rotate:
            grid = np.mgrid[0:w, 0:h].T
            dfc = grid-[cx, cy]

        # zooming, spirals/rotations
        if to_zoom or to_spiral or to_rotate:
            dfc2 = np.power(dfc, 2)
            dst = np.power(np.sum(dfc2, axis=2), 0.5)
            ang = np.arctan2(dfc[:,:,1], dfc[:,:,0])
            dst2, ang2 = dst, ang

        # xy stretch/compress/shift
        if to_shift:
            stretch_mul = 1.0 / stretch[0], 1.0 / stretch[1]
            sy = cy - h * shift[0] + dfc[:,:,1] * stretch_mul[0]
            sx = cx - w * shift[1] + dfc[:,:,0] * stretch_mul[1]
            idx2 = np.dstack([sy, sx])
            mod_idxs[idxm][0] = idx2

        # expand/contract, plain zoom + radial
        if to_zoom:
            dst2 = np.add(np.multiply(1.0/zoom, dst), -expand).clip(min=0)

        # more complex dist multiply
        if to_spiral:
            dst2 = np.multiply(dst2, 1.0 + spiral_margin * np.sin(spiral_periods * ang))

        # rotation
        if to_rotate:
            ang2 = np.add(ang, rot_const + rot_ang * ang + rot_dst * dst)

        # re-map according to dst2 and ang2
        if to_zoom or to_spiral or to_rotate:
            idx_a = ang2[grid[:,:,1], grid[:,:,0]]
            idx_d = dst2[grid[:,:,1], grid[:,:,0]]
            idx_sin = cy + np.multiply(idx_d, np.sin(idx_a))
            idx_cos = cx + np.multiply(idx_d, np.cos(idx_a))
            idx2 = np.dstack((np.array(idx_sin), np.array(idx_cos)))
            mod_idxs[idxm][1] = idx2

        # perlin noise, very inefficient because of the double for loop
        if to_noise:
            offyy, offyx, offxy, offxx = noise_offset[1], noise_offset[1], noise_offset[0], noise_offset[0]
            nyy, nyx, nxy, nxx = noise_rate[1], noise_rate[1], noise_rate[0], noise_rate[0]
            midy, marginy = 0.5*(-noise_margin[0]+noise_margin[0]), 0.5*(noise_margin[0]--noise_margin[0])
            midx, marginx = 0.5*(-noise_margin[1]+noise_margin[1]), 0.5*(noise_margin[1]--noise_margin[1])
            idx2 = [[(y+midy+marginy*snoise2(offyy+nyy*y/h, offyx+nyx*x/w, 3), 
                      x+midx+marginx*snoise2(offxy+nxy*y/h, offxx+nxx*x/w, 3)) 
                    for x in range(w)] for y in range(h)]
            mod_idxs[idxm][2] = idx2

    # average shift-indexes and transform the image
    for i in range(len(mod_idxs[0])):
        idxs = [np.multiply(mod_idxs[m][i], np.expand_dims(masks[:,:,m], 2)) for m in range(len(mods))]
        idx2 = np.sum(idxs, axis=0)
        img = map_image(img, idx2)

    if to_pil:
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    
    return img






# def inject_image(img0, path, amt, matchHist=True):
#     hist0 = get_histogram(img0.astype('uint8'), bright=False)
#     #img1 = scipy.misc.imread(path, mode='RGB')
#     img1 = np.array(Image.open(path).convert('RGB'))
#     (h, w), (ih, iw) = (img0.shape[0:2]), (img1.shape[0:2])
#     if float(w)/h > float(iw)/ih:
#         d = ih - iw * float(h) / w
#         if d>0:
#             img1 = img1[int(d/2):-int(d/2),:,:]
#     else:
#         d = iw - ih * float(w) / h
#         if d>0:
#             img1 = img1[:, int(d/2):-int(d/2),:]
#     img1 = resize(img1, (h, w))
#     img2 = (1.0 - amt) * img0 + amt * img1
#     if matchHist:
#         img2 = match_histogram(img2, hist0)
#     return img2.astype('float32')    
