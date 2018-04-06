import os
from os import listdir
from os.path import isfile, join

from PIL import Image
# args
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



mypath = "../datasets/landscape"
mypath2 = "../datasets/landscape_128_128"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) ]
for o in onlyfiles:
	try:
		im = Image.open(join(mypath,o)).convert("RGB")
		im = im.resize((128,128), Image.BICUBIC)
		print(im)
		im.save(join(mypath2,o))
	except:
		print('error...')
		
