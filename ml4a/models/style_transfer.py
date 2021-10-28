from ..utils import downloads
from .. import image
from ..utils import EasyDict
from . import submodules

import os
import io
from pathlib import Path
from PIL import Image, ImageCms
import torch

cuda_available = submodules.cuda_available()

srgb_profile_file = Path(os.path.join(submodules.get_submodules_root('style-transfer-pytorch'), 'style_transfer/sRGB Profile.icc'))
srgb_profile = srgb_profile_file.read_bytes()

model = None

with submodules.localimport('submodules/style-transfer-pytorch/style_transfer') as _importer:
    from style_transfer import STIterate, StyleTransfer


def setup_styletransfer(devices, pooling):
    global model
    model = StyleTransfer(devices=devices, pooling=pooling)
    return model


def preprocess_image(img, size):
    src_prof = dst_prof = srgb_profile
    img = image.load_image(img, size)
    if 'icc_profile' in img.info:
        src_prof = img.info['icc_profile']
    else:
        img = img.convert('RGB')
    if src_prof == dst_prof:
        img = img.convert('RGB')
    src_prof = io.BytesIO(src_prof)
    dst_prof = io.BytesIO(dst_prof)
    img = ImageCms.profileToProfile(img, src_prof, dst_prof, outputMode='RGB')
    return img
    

def run(config, img=None, title=None):
    config = EasyDict(config)
    config.style_weights = config.style_weights if 'style_weights' in config else None
    config.content_weight = float(config.content_weight) if 'content_weight' in config else 0.015
    config.tv_weight = config.tv_weight if 'tv_weight' in config else 2.0
    config.min_scale = config.min_scale if 'min_scale' in config else 128
    config.size = config.size if 'size' in config else None
    config.end_scale = max(config.size) if isinstance(config.size, tuple) else config.size
    config.pooling = config.pooling if 'pooling' in config else 'max'
    config.iterations = config.iterations if 'iterations' in config else 500
    config.initial_iterations = config.initial_iterations if 'initial_iterations' in config else 1000
    config.step_size = config.step_size if 'step_size' in config else 0.02
    config.avg_decay = config.avg_decay if 'avg_decay' in config else 0.99
    config.init = config.init if 'init' in config else 'content'
    config.style_scale_fac = config.style_scale_fac if 'style_scale_fac' in config else 1.0
    config.style_size = config.style_size if 'style_size' in config else None
    config.devices = config.devices if 'devices' in config else [0]
    config.devices = config.gpu if 'gpu' in config else config.devices
    config.style_images = config.style_image if 'style_image' in config else config.style_images
    config.style_images = config.style_images if isinstance(config.style_images, list) else [config.style_images]
    assert config.pooling in ['max', 'average', 'l2'], 'error: pooling method not recognized'

    init_img = preprocess_image(img, config.size) if img is not None else None
    content_img = preprocess_image(config.content_image, config.size)
    style_imgs = [preprocess_image(img, config.size) for img in config.style_images]
    devices = [torch.device('cuda:%d'%g) for g in config.gpus]
    
    del config.size
    for device in devices:
        torch.tensor(0).to(device)

    if model is None:
        setup_styletransfer(config.devices, config.pooling)
    
    defaults = StyleTransfer.stylize.__kwdefaults__
    st_kwargs = {k: v for k, v in config.__dict__.items() if k in defaults}
    model.stylize(init_img, content_img, style_imgs, **st_kwargs, callback=None)

    output_image = model.get_image('pil')
    return output_image
