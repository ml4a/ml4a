from pathlib import Path
from imageio import imread, imwrite
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

#with submodules.import_from('PhotoSketch'):  # localimport fails here   
with submodules.localimport('submodules/FlowNetPytorch') as _importer:
    import models
    import flow_transforms
    from util import flow2rgb

    
model = None


def setup():
    global model, device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_names = sorted(name for name in models.__dict__ 
                         if name.islower() and not name.startswith("__"))
    
    model_path = downloads.download_from_gdrive(
        '1jbWiY1C_nqAUJRYZu7mwzV6CK7ugsa5v', 
        'FlowNetPytorch/flownets_EPE1.951.pth.tar')

    # this model may be better, but requires spatial-correlation-sampler, which fails to install with pip
    #model_path = downloads.download_from_gdrive(
    #      '1H_5WE-Lrx5arD0-X801yRzdSAuBZQmXh', 
    #      'FlowNetPytorch/flownetc_EPE1.766.tar')
    
    network_data = torch.load(model_path)
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True
    if 'div_flow' in network_data.keys():
        div_flow = network_data['div_flow']


def flow_to_mapping(flow):
    h, w = flow.shape[:2]
    grid = np.mgrid[0:w, 0:h].T
    distort = grid + flow
    distort = distort[:,:,[1,0]]
    return distort


def blur(flow, blur_times=1):
    for f in range(blur_times):
        flow = cv2.GaussianBlur(flow, (3, 3), 0)
    return flow


def run(img1, img2, bidirectional=False, upsampling='bilinear', div_flow=20, max_flow=None, to_rgb=False):
    if model is None:
        setup()
        
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img1 = input_transform(np.array(img1))
    img2 = input_transform(np.array(img2))
    input_var = torch.cat([img1, img2]).unsqueeze(0)

    if bidirectional:
        inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
        input_var = torch.cat([input_var, inverted_input_var])

    input_var = input_var.to(device)
    flow = model(input_var)

    if upsampling is not None:
        assert upsampling in ['nearest', 'bilinear'], \
            'Upsampling mode {} not recognized'.format(upsampling)
        flow = torch.nn.functional.interpolate(
            flow, 
            size=img1.size()[-2:], 
            mode=upsampling, 
            align_corners=False)

    if to_rgb:
        rgb_flow = flow2rgb(div_flow * flow[0], max_value=max_flow)
        rgb_flow = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
        return rgb_flow
    else:
        flow = (div_flow * flow[0]).detach().cpu().numpy().transpose(1,2,0)
        return flow
