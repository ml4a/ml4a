import os
import csv
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from scipy.io import loadmat
from distutils.version import LooseVersion

from .. import image
from . import submodules
from ..utils import downloads

cuda_available = submodules.cuda_available()

#with submodules.import_from('semantic-segmentation-pytorch'):
with submodules.localimport('submodules/semantic-segmentation-pytorch') as _importer:
    from mit_semseg.dataset import TestDataset
    from mit_semseg.models import ModelBuilder, SegmentationModule
    from mit_semseg.utils import colorEncode, find_recursive, setup_logger
    from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
    from mit_semseg.lib.utils import as_numpy
    #from mit_semseg.config import cfg


segmentation_module = None

color_path = downloads.download_data_file(
    'https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/color150.mat', 
    'semantic-segmentation-pytorch/data/color150.mat'
)
data_path = downloads.download_text_file(
    'https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/data/object150_info.csv', 
    'semantic-segmentation-pytorch/data/object150_info.csv'
)

# colors and class names
colors = loadmat(color_path)['colors']
classes = {}
with open(data_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        classes[int(row[0])-1] = row[5].split(";")[0]


def setup(gpu):
    global segmentation_module

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights=downloads.download_data_file(
            'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth', 
            'semantic-segmentation-pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth'
        )
    )
    
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights=downloads.download_data_file(
            'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth', 
            'semantic-segmentation-pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth'
        ),
        use_softmax=True
    )

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()

    
def get_class_index(class_name):
    if class_name not in classes.values():
        return None
    return list(classes.keys())[list(classes.values()).index(class_name)]


def get_color_labels(pred):
    im_vis = colorEncode(pred, colors).astype(np.uint8)
    return im_vis
    
    
def visualize(img, pred, index=None, concat_original=True):
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1        
    im_vis = colorEncode(pred, colors).astype(np.uint8)
    if concat_original:
        im_vis = np.concatenate((img, im_vis), axis=1)
    image.display(Image.fromarray(im_vis))
        

def get_mask(pred, index):
    is_list = isinstance(index, list)
    index = index if is_list else [index]
    index = [get_class_index(idx) if isinstance(idx, str) else idx 
             for idx in index]
    h, w = pred.shape[:2]
    mask = np.zeros((h, w, len(index)))
    for i, idx in enumerate(index):
        mask_channel = pred.copy()
        mask_channel[pred != idx] = 0
        mask_channel[pred == idx] = 255
        mask[:, :, i] = mask_channel
    if len(index) == 1:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)
    
    
def run(imgs, gpu=0):
    if segmentation_module is None:
        setup(gpu)
    
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    img_data = pil_to_tensor(imgs)
    singleton_batch = {'img_data': img_data[None].cuda()}
    
    # Run the segmentation at the highest resolution
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=img_data.shape[1:])

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return pred
        
