import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

#with submodules.localimport('submodules/PhotoSketch') as _importer:
with submodules.import_from('PhotoSketch'):  # localimport fails here   
    from models.models import create_model
    import util.util as util


model = None


def setup_photosketch():
    global model
    model_directory = downloads.download_from_gdrive(
        '1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj', 
        'photosketch/pretrained',
        zip_file=True)
    opt = {}
    opt = SimpleNamespace(**opt)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True 
    opt.name = model_directory
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
    opt.pretrain_path = model_directory
    model = create_model(opt)
    return model


def run(img):
    if model is None:
        setup_photosketch()
    img = np.array(img) / 255.0
    h, w = img.shape[0:2]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float() #.to(device)
    data = {'A_paths': '', 'A': img, 'B': img }
    model.set_input(data)
    model.test()
    output = util.tensor2im(model.fake_B)
    return output
