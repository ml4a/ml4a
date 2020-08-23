import numpy as np
import torch
from PIL import Image
#import options.test_options 
from models.models import create_model
import util.util as util
from types import SimpleNamespace
from dataset_utils import cv2pil, pil2cv


def setup(model_dir):
    global model
    opt = {}
    opt = SimpleNamespace(**opt)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True 
    opt.name = 'pretrained'
    opt.checkpoints_dir = '.'
    opt.model = 'pix2pix'
    opt.which_direction = 'AtoB'
    opt.norm = 'batch'
    opt.input_nc = 3
    opt.output_nc = 1
    opt.which_model_netG = 'resnet_9blocks'
    opt.no_dropout = True
    opt.isTrain = False
    opt.use_cuda = True
    opt.ngf = 64
    opt.ndf = 64
    opt.init_type = 'normal'
    opt.which_epoch = 'latest'
    opt.pretrain_path = model_dir
    model = create_model(opt)
    return model


def sketch(img):
    img = pil2cv(img) # cv2.imread(image_path)
    img = img / 255.
    h, w = img.shape[0:2]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float() #.to(device)
    data = {'A_paths': '', 'A': img, 'B': img }
    model.set_input(data)
    model.test()
    output = util.tensor2im(model.fake_B)
    return cv2pil(output)
