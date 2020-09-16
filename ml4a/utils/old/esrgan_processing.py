from collections import OrderedDict
import numpy as np
import torch
import architecture as arch
from dataset_utils import cv2pil, pil2cv

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = None


def setup(model_dir):
    global model
    alpha=0.5
    net_PSNR_path = '%s/RRDB_PSNR_x4.pth' % model_dir
    net_ESRGAN_path = '%s/RRDB_ESRGAN_x4.pth' % model_dir
    net_PSNR = torch.load(net_PSNR_path)
    net_ESRGAN = torch.load(net_ESRGAN_path)
    net_interp = OrderedDict()
    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(net_interp)
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model

def upsample(img):
    img = pil2cv(img) # cv2.imread(image_path)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()    
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype('uint8')
    return cv2pil(output)
