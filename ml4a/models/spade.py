import os
import sys
import numpy as np
from PIL import Image

from ..utils import EasyDict
from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

#with submodules.localimport('submodules/SPADE') as _importer:
with submodules.import_from('SPADE'):  # localimport fails here    
    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from options.base_options import BaseOptions
    from data.base_dataset import get_params, get_transform
    import util.util as util


pretrained_models = {
    'cityscapes': [
        ['1_APZIT-3eD8KXK6GFz4cFXpN2qlQbez1', 'latest_net_G.pth'],
        ['1zIxWGADWABWQRdXZqTWlzg7RwHfWZLt9', 'opt.txt'],
        ['1Wn-OAagSYZplZJusvK9In8WtqwjS_ij2', 'classes_list.txt']
    ],
    'ade20k': [
        ['1shvEumc5PrqXIahV61_fLRNOUTb96wSg', 'latest_net_G.pth'],
        ['1JbjQj7AdHgFrCRQFPTADybLS7AzyoyJ1', 'opt.txt'],
        ['1FQ59iTkQ3fSnjEuaYsn7WdBB_j1fAOhE', 'classes_list.txt']
    ],
    'coco': [
        ['16KfJKje4aNUQSAxmzzKowJxpPzYgUOlo', 'latest_net_G.pth'],
        ['1Ed16m6SAZNoQwSA2-fGYOhqWWU671M47', 'opt.txt'],
        ['1XukXJvb2tYbEcvSCenRDFHkypdUJUj1Q', 'classes_list.txt']
    ],
    'landscapes': [
        ['1T9FGxZQL9riB-a-cBOkFDdjBC2rf1Buh', 'latest_net_G.pth'],
        ['1kgqOc4mlvOt1gjxrCchnSyzp_6NgJ6wX', 'opt.txt'],
        ['1jsgr-6TZHDFll9ZdszpY8JNyY6B_5MzI', 'classes_list.txt']
    ]
}

    
classes = None
model = None
opt = None


def get_pretrained_models():
    return pretrained_models.keys()


def parse_opt_file(path):
    file = open(path, 'rb')
    opt = {}
    for line in file.readlines():
        line = str(line).split(': ')
        key = line[0].split(' ')[-1]
        value = line[1].split(' ')[0]
        opt[key] = value
    return opt


def get_class_index(class_name):
    if class_name not in classes.values():
        return None
    return list(classes.keys())[list(classes.values()).index(class_name)]


def load_model(model_name):
    assert model_name in pretrained_models, \
        '{} not recongized. Available models: {}'.format(model_name, ', '.join(pretrained_models))
    
    global model, opt, classes
    
    model_subfolder = os.path.join('SPADE/checkpoints/', model_name)
    checkpoint_dir = os.path.join(downloads.get_ml4a_downloads_folder(), model_subfolder)
    all_checkpoints_dir = os.path.join(downloads.get_ml4a_downloads_folder(), 'SPADE/checkpoints/')
    
    for gdrive_id, filename in pretrained_models[model_name]:
        location = os.path.join(model_subfolder, filename)
        downloads.download_from_gdrive(gdrive_id, location)

    with open(os.path.join(checkpoint_dir, 'classes_list.txt'), 'r') as classes_file:
        classes = eval(classes_file.read())
    
    opt_file = os.path.join(checkpoint_dir, 'opt.txt')
    parsed_opt = parse_opt_file(opt_file)
    
    opt = EasyDict({})
    opt.isTrain = False
    opt.checkpoints_dir = all_checkpoints_dir
    opt.name = model_name
    opt.aspect_ratio = float(parsed_opt['aspect_ratio'])
    opt.load_size = int(parsed_opt['load_size'])
    opt.crop_size = int(parsed_opt['crop_size'])
    opt.no_instance = True if parsed_opt['no_instance']=='True' else False
    opt.preprocess_mode = parsed_opt['preprocess_mode']
    opt.contain_dontcare_label = True if parsed_opt['contain_dontcare_label']=='True' else False
    opt.gpu_ids = parsed_opt['gpu_ids']
    opt.netG = parsed_opt['netG']
    opt.ngf = int(parsed_opt['ngf'])
    opt.num_upsampling_layers = parsed_opt['num_upsampling_layers']
    opt.use_vae = True if parsed_opt['use_vae']=='True' else False  
    opt.label_nc = int(parsed_opt['label_nc'])
    opt.semantic_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
    opt.norm_G = parsed_opt['norm_G']
    opt.init_type = parsed_opt['init_type']
    opt.init_variance = float(parsed_opt['init_variance'])
    opt.which_epoch = parsed_opt['which_epoch']
    model = Pix2PixModel(opt)
    model.eval()
    
    
def run(labelmap):
    assert model is not None, 'Error: no model loaded.'
    labelmap = Image.fromarray(np.array(labelmap).astype(np.uint8))
    params = get_params(opt, labelmap.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(labelmap) * 255.0
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