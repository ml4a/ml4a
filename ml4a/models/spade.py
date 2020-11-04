import os
import sys
import numpy as np
from PIL import Image



from ..utils import EasyDict
from ..utils import downloads
from . import submodules

with submodules.import_from('SPADE'):
    print("imp  1 spade")
    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from options.base_options import BaseOptions
    from data.base_dataset import get_params, get_transform
    import util.util as util
    print("imp 2 spade")
    

model = None
opt = None


def parse_opt_file(path):
    file = open(path, 'rb')
    opt = {}
    for line in file.readlines():
        line = str(line).split(': ')
        key = line[0].split(' ')[-1]
        value = line[1].split(' ')[0]
        opt[key] = value
    return opt


def setup():
    global model, opt #, get_params, get_transform, util

    checkpoints_dir = 'SPADE/checkpoints/'
    checkpoint_name = 'Labels2Landscapes_512'
    landscapes_dir = os.path.join(checkpoints_dir, checkpoint_name)
    
    landscape_files = [
        ['1CXk6QPKeGLgp_VZwSUJFqnsvDuixx2fr', os.path.join(landscapes_dir, 'iter.txt')],
        ['1tsfDW8xb_Vat3En3hqVmoAQQBp8umNdV', os.path.join(landscapes_dir, 'latest_net_D.pth')],
        ['1T9FGxZQL9riB-a-cBOkFDdjBC2rf1Buh', os.path.join(landscapes_dir, 'latest_net_G.pth')],
        ['17dLGaO0l2oiAp7QVopON-yXw8DDs-M0e', os.path.join(landscapes_dir, 'loss_log.txt')],
        ['1b9n6RQN6GaSY8cvyZewM0xbNbAzJzGEF', os.path.join(landscapes_dir, 'opt.pkl')],
        ['1kgqOc4mlvOt1gjxrCchnSyzp_6NgJ6wX', os.path.join(landscapes_dir, 'opt.txt')]
    ]
    print(landscape_files)

    for gdrive_id, location in landscape_files:
        downloads.download_from_gdrive(gdrive_id, location)

     
    
    
    checkpoints_dir = os.path.join(downloads.get_ml4a_downloads_folder(), checkpoints_dir)
    landscapes_dir = os.path.join(downloads.get_ml4a_downloads_folder(), landscapes_dir)

    opt_file = os.path.join(landscapes_dir, 'opt.txt')
    
    parsed_opt = parse_opt_file(opt_file)
    
    
    
    opt = EasyDict({})
    
    
    
    
    opt.isTrain = False
    opt.checkpoints_dir = checkpoints_dir
    opt.name = checkpoint_name
    opt.aspect_ratio = float(parsed_opt['aspect_ratio'])
    opt.load_size = int(parsed_opt['load_size'])
    opt.crop_size = int(parsed_opt['crop_size'])
    opt.label_nc = int(parsed_opt['label_nc'])
    opt.no_instance = True if parsed_opt['no_instance']=='True' else False
    opt.preprocess_mode = parsed_opt['preprocess_mode']
    opt.contain_dontcare_label = True if parsed_opt['contain_dontcare_label']=='True' else False
    opt.gpu_ids = parsed_opt['gpu_ids']
    opt.netG = parsed_opt['netG']
    opt.ngf = int(parsed_opt['ngf'])
    opt.num_upsampling_layers = parsed_opt['num_upsampling_layers']
    opt.use_vae = True if parsed_opt['use_vae']=='True' else False  
    opt.semantic_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
    opt.norm_G = parsed_opt['norm_G']
    opt.init_type = parsed_opt['init_type']
    opt.init_variance = float(parsed_opt['init_variance'])
    opt.which_epoch = parsed_opt['which_epoch']

    model = Pix2PixModel(opt)
    model.eval()
    
    
    
    
    
def run(image):
    if model is None:
        setup()
    
    print("RUN!@@@2222")
    
#    image = np.ones((512, 512)).astype(np.uint8)
    image = Image.fromarray(np.array(image).astype(np.uint8))
    print('udhsdg')
    colors = [[255,255,255], [255,0,0], [0,255,0], [0,0,255], [0,0,0]]
    #colors = [s[1] for s in shapes]
    #labels = [s[2] for s in shapes]
    #img = image2colorlabels(img, colors, labels)
    params = get_params(opt, image.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(image) * 255.0
    print("lbales", opt.label_nc)
    label_tensor[label_tensor == 255.0] = opt.label_nc
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(Image.new('RGB', (500, 500)))
    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    output = util.tensor2im(generated[0])
    output = Image.fromarray(output)
    return output