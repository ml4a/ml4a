import os
import sys
import numpy as np
from PIL import Image
from localimport import localimport

from ..utils import EasyDict
from ..utils import downloads
from . import submodules

#with localimport('submodules/SPADE') as _importer:
with submodules.import_from('SPADE'):  # localimport fails here    
    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from options.base_options import BaseOptions
    from data.base_dataset import get_params, get_transform
    import util.util as util
    
    
classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'street sign', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'hat', 26: 'backpack', 27: 'umbrella', 28: 'shoe', 29: 'eye glasses', 30: 'handbag', 31: 'tie', 32: 'suitcase', 33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite', 38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard', 42: 'tennis racket', 43: 'bottle', 44: 'plate', 45: 'wine glass', 46: 'cup', 47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'couch', 63: 'potted plant', 64: 'bed', 65: 'mirror', 66: 'dining table', 67: 'window', 68: 'desk', 69: 'toilet', 70: 'door', 71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 76: 'cell phone', 77: 'microwave', 78: 'oven', 79: 'toaster', 80: 'sink', 81: 'refrigerator', 82: 'blender', 83: 'book', 84: 'clock', 85: 'vase', 86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush', 90: 'hair brush', 91: 'banner', 92: 'blanket', 93: 'branch', 94: 'bridge', 95: 'building-other', 96: 'bush', 97: 'cabinet', 98: 'cage', 99: 'cardboard', 100: 'carpet', 101: 'ceiling-other', 102: 'ceiling-tile', 103: 'cloth', 104: 'clothes', 105: 'clouds', 106: 'counter', 107: 'cupboard', 108: 'curtain', 109: 'desk-stuff', 110: 'dirt', 111: 'door-stuff', 112: 'fence', 113: 'floor-marble', 114: 'floor-other', 115: 'floor-stone', 116: 'floor-tile', 117: 'floor-wood', 118: 'flower', 119: 'fog', 120: 'food-other', 121: 'fruit', 122: 'furniture-other', 123: 'grass', 124: 'gravel', 125: 'ground-other', 126: 'hill', 127: 'house', 128: 'leaves', 129: 'light', 130: 'mat', 131: 'metal', 132: 'mirror-stuff', 133: 'moss', 134: 'mountain', 135: 'mud', 136: 'napkin', 137: 'net', 138: 'paper', 139: 'pavement', 140: 'pillow', 141: 'plant-other', 142: 'plastic', 143: 'platform', 144: 'playingfield', 145: 'railing', 146: 'railroad', 147: 'river', 148: 'road', 149: 'rock', 150: 'roof', 151: 'rug', 152: 'salad', 153: 'sand', 154: 'sea', 155: 'shelf', 156: 'sky-other', 157: 'skyscraper', 158: 'snow', 159: 'solid-other', 160: 'stairs', 161: 'stone', 162: 'straw', 163: 'structural-other', 164: 'table', 165: 'tent', 166: 'textile-other', 167: 'towel', 168: 'tree', 169: 'vegetable', 170: 'wall-brick', 171: 'wall-concrete', 172: 'wall-other', 173: 'wall-panel', 174: 'wall-stone', 175: 'wall-tile', 176: 'wall-wood', 177: 'water-other', 178: 'waterdrops', 179: 'window-blind', 180: 'window-other', 181: 'wood'}

model = None
opt = None


def get_class_index(class_name):
    if class_name not in classes.values():
        return None
    return list(classes.keys())[list(classes.values()).index(class_name)]


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
    
    
def run(labelmap):
    if model is None:
        setup()
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