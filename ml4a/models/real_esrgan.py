import os
import numpy as np
import torch

from ..utils import downloads
from . import submodules
from .. import image

cuda_available = submodules.cuda_available()


#with submodules.import_from('Real-ESRGAN'):  # localimport fails here   
with submodules.localimport('submodules/Real-ESRGAN') as _importer:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = None
upsampler = None
netscale = None
outscale = None


def setup():
    global model, upsampler
    global netscale, outscale
    
    model_path = downloads.download_data_file(
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
        os.path.join('Real_ESRGAN', 'RealESRGAN_x4plus.pth'))
    
    netscale = 4
    outscale = 4
            
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=netscale)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False)

    
def run(img):
    if model is None:
        setup()
        
    if isinstance(img, str):
        img = image.load_image(img)
        
    w, h = image.get_size(img)
    if max(h, w) > 1000 and netscale == 4:
        import warnings
        warnings.warn('The input image is large, try X2 model for better performance.')
    if max(h, w) < 500 and netscale == 2:
        import warnings
        warnings.warn('The input image is small, try X4 model for better performance.')

    #_, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    output, _ = upsampler.enhance(np.array(img), outscale=outscale)
    return output
    
