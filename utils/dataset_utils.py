import os
from os import listdir
from os.path import isfile, join
from random import random
from PIL import Image
import argparse
from tqdm import tqdm

#args
# input_dir source directory of images
# output_dir 
# output_format rgb, g
# output_size [w, h] or keep the same
# crops/centers
#  - to crop or to hard-resize
# duplication + augmentation (sheer, rotate, etc)
# output save format (jpg png)


# todo
# make recursive?

parser = argparse.ArgumentParser()
parser.add_argument("--frac", type=float, help="cropping ratio before resizing", default=0.6667)
parser.add_argument("--w", type=int, help="output image width", default=64)
parser.add_argument("--h", type=int, help="output image width", default=64)
parser.add_argument("--input_dir", help="where to get input images")
parser.add_argument("--output_dir", help="where to put output images")


def crop_resize(im1, frac, w2, h2):
	ar = float(w2 / h2)
	h1, w1 = im1.height, im1.width
	if float(w1 / h1) > ar:
		h1_crop = max(h2, h1 * frac)
		w1_crop = h1_crop * ar
	else:
		w1_crop = max(w2, w1 * frac)
		h1_crop = w1_crop / ar
	x_crop, y_crop = (w1 - w1_crop - 1) * random(), (h1 - h1_crop - 1) * random()
	h1_crop, w1_crop, y_crop, x_crop = int(h1_crop), int(w1_crop), int(y_crop), int(x_crop)
	im1_crop = im1.crop((x_crop, y_crop, x_crop+w1_crop, y_crop+h1_crop))
	im2 = im1_crop.resize((w2, h2), Image.BICUBIC)
	return im2

def main(input_dir, output_dir, w2, h2, frac):
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	images = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) ]
	for img_path in tqdm(images):
		try:
			im1 = Image.open(join(input_dir, img_path)).convert("RGB")
			im2 = crop_resize(im1, frac, w2, h2)
			im2.save(join(output_dir, img_path))
		except:
			print('error...')

if __name__ == '__main__':
	args = parser.parse_args()
	w2, h2, frac = args.w, args.h, args.frac
	input_dir, output_dir = args.input_dir, args.output_dir
	main(input_dir, output_dir, w2, h2, frac)

