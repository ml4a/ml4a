import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable

from ..utils import downloads
from .. import image
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/BASNet') as _importer:
    from data_loader import RescaleT, ToTensorLab
    from model import BASNet

net = None
model_loaded = False


def load_model(model_dir):
    net = BASNet(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    return net


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
        
    
def get_foreground(img):
    
    global model_loaded, net
    if not model_loaded:
        basnet_model_file = downloads.download_from_gdrive(
            gdrive_fileid='1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu', 
            output_path='BASNet/saved_models/basnet_bsi/basnet.pth')
        net = load_model(basnet_model_file)
        model_loaded = True
    
    img = np.array(img)
    size = image.get_size(img)
    
    label = np.zeros(img.shape)
    sample = {'image':img, 'label': label}

    transform = transforms.Compose([RescaleT(256),ToTensorLab(flag=0)])
    sample = transform(sample)

    inputs_test = sample['image']
    inputs_test = inputs_test.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
    inputs_test = inputs_test.unsqueeze(0)

    d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    del d1,d2,d3,d4,d5,d6,d7,d8

    pred = pred.squeeze()
    pred_np = pred.cpu().data.numpy()

    im_mask = Image.fromarray(pred_np * 255).convert('RGB')
    im_mask = im_mask.resize(size, resample=Image.BILINEAR)
    im_mask = np.array(im_mask)
    
    return im_mask

    
