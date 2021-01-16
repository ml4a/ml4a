import os
import numpy as np
import torch

from ..utils import downloads
from . import submodules
from .. import image

cuda_available = submodules.cuda_available()

#with submodules.import_from('ESRGAN'):  # localimport fails here   
with submodules.localimport('submodules/ESRGAN') as _importer:
    import RRDBNet_arch as arch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = None


def setup():
    global model
    
    model_path = downloads.download_from_gdrive(
        '1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene', 
        os.path.join('ESRGAN', 'RRDB_ESRGAN_x4.pth'))
    
#     model_path = downloads.download_from_gdrive(
#         '1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN', 
#         os.path.join('ESRGAN', 'RRDB_PSNR_x4.pth'))

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    
def run(img):
    if model is None:
        setup()
        
    if isinstance(img, str):
        img = image.load_image(img)

    img = np.array(img)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output
