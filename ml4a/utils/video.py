from PIL import Image, ImageStat, ImageEnhance
import sys, argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
from tqdm import tqdm

from .console import *
from .histogram import *

ImageEnhance.LOAD_TRUNCATED_IMAGES = True
ImageStat.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True


def generate_video2(frames_path, output_path):
    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f))])
    log('loading %d images from %s' % (numframes, frames_path))
    images = [ Image.open('%s/f%05d.png'%(frames_path, t+1)) for t in range(numframes) ]
    brightness = [ ImageStat.Stat(img.convert('L')).mean[0] for img in images ]
    avg_brightness = np.mean(brightness)
    os.system('mkdir %s/temp' % frames_path)
    for i, img in tqdm(enumerate(images)):
        mult = avg_brightness / brightness[i]
        source = img.split()
        r = source[0].point(lambda i: i*mult)
        g = source[1].point(lambda i: i*mult)
        b = source[2].point(lambda i: i*mult)
        img2 = Image.merge(img.mode, (r, g, b))
        img2 = ImageEnhance.Contrast(img2).enhance(1.2)
        img2 = ImageEnhance.Sharpness(img2).enhance(1.25)
        img2.save("%s/temp/f%05d.png" % (frames_path, i+1))
    w, h  = images[0].size
    cmd = 'ffmpeg -i %s/temp/f%%05d.png -c:v libx264 -pix_fmt yuv420p -vf scale=%d:%d %s'%(frames_path, w-(w%2), h-(h%2), output_path)
    log('creating movie at %s' % output_path)
    os.system('rm %s' % output_path)
    os.system(cmd)
    os.system('rm -rf %s/temp' % frames_path)

    
def generate_video(frames_path, output_path, sat=1.0, con=1.0, sharp=1.0, match_hist=False, bitrate=None, cumulative=False, erase_frames=True):
    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f)) and f[-4:]=='.png'])
    if numframes == 0:
        #warn("No frames found in %s"%frames_path)
        print("No frames found in %s"%frames_path)
        return
    log('creating %d-frame movie: %s -> %s'%(numframes, frames_path, output_path))
    if match_hist:
        avg_hist = get_average_histogram(frames_path)
    os.system('mkdir %s/temp' % frames_path)
    for i in tqdm(range(numframes)):
        edited_filepath = "%s/temp/f%05d.png" % (frames_path, i+1)
        if not os.path.isfile(edited_filepath) or not cumulative:
            img = Image.open('%s/f%05d.png'%(frames_path, i+1))
            if match_hist:
                img = match_histogram(img, avg_hist)
            img = ImageEnhance.Color(img).enhance(sat)
            img = ImageEnhance.Contrast(img).enhance(con)
            img = ImageEnhance.Sharpness(img).enhance(sharp)
            img.save(edited_filepath)
    w, h = img.size
    wx, wy = w-(w%2), h-(h%2)
    ffmpeg_str = 'ffmpeg -i %s/temp/f%%05d.png -c:v libx264 -pix_fmt yuv420p -vf scale=%d:%d ' % (frames_path, wx, wy)
    if bitrate is None:
        cmd = '%s %s'%(ffmpeg_str, output_path)
    else:
        cmd = '%s -b %d %s'%(ffmpeg_str, bitrate, output_path)
    os.system(cmd)
    #if erase_frames:
    #    os.system('rm %s' % output_path)
    if not cumulative:
        os.system('rm -rf %s/temp' % frames_path)
    log('Done making %s' % output_path)


def gen_video(frames_path, sat, con, sharp, bitrate=None, cumulative=False):
    output_path = '%s_%0.2f,%0.2f,%0.2f.mp4'%(frames_path, sat, con, sharp)
    generate_video(frames_path, output_path, sat, con, sharp, bitrate, cumulative)

    
def gen_video_dir(frames_path, sat, con, sharp, overwrite=True, bitrate=None, cumulative=False):
    dirs = [f for f in listdir(frames_path) if isdir(join(frames_path, f))]
    for d in dirs:
        output_path = '%s/%s_%0.2f,%0.2f,%0.2f.mp4'%(frames_path, d, sat, con, sharp)
        file_exists = os.path.isfile(output_path)
        if overwrite or not file_exists:
            log("generate %s"%output_path)
            gen_video(join(frames_path, d), sat, con, sharp, bitrate, cumulative)


def process_arguments(args):
    parser = argparse.ArgumentParser(description='generate video')
    parser.add_argument('--video', action='store', help='path to directory of input video')
    parser.add_argument('--dir', action='store', help='path to directory of directories of input videos')
    parser.add_argument('--sacosh', action='store', help='saturation, contrast, sharpness')
    params = vars(parser.parse_args(args))
    return params


def main():
    params = process_arguments(sys.argv[1:])
    sa, co, sh = 1.0, 1.0, 1.0
    if params['sacosh'] is not None:
        sacosh = params['sacosh'].split(',')
        sa, co, sh = float(sacosh[0]), float(sacosh[1]), float(sacosh[2])
    if params['video'] is not None:
        gen_video(params['video'], sa, co, sh)
    elif params['dir'] is not None:
        gen_video_dir(params['dir'], sa, co, sh, overwrite=True)
    

    
    

    
##############

from tqdm import tqdm
import cv2


# def write_from_frames

def write_from_generator(video_name, latents, generator_function, fps=30, batch_size=16):
    num_frames = latents.shape[0]
    for f in tqdm(range(0, num_frames, batch_size)):
        f1, f2 = f, min(num_frames-1, f+batch_size)
        batch = latents[f1:f2, :]
        imgs = generator_function(batch)
        if f == 0:
            height, width = imgs[0].shape[0:2]
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=30, frameSize=(width, height))
        for iter in range(0, imgs.shape[0]):
            video.write(imgs[iter,:,:,::-1])
    cv2.destroyAllWindows()
    video.release()
    
    
#############
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
