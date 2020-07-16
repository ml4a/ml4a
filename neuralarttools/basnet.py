from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable

from .util import *

net = None
model_loaded = False


def load_model(model_dir):
    from model import BASNet
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


def setup_basnet(basnet_path, basnet_model_path):
    import sys
    sys.path.append(basnet_path)
    
    from data_loader import RescaleT
    from data_loader import ToTensorLab

    global model_loaded, net
    if not model_loaded:
        net = load_model(basnet_model_path)
        model_loaded = True

        
def get_foreground(img, basnet_path=None, basnet_model_path=None):
    if basnet_path is not None and basnet_model_path is not None:
        setup_basnet(basnet_path, basnet_model_path)
    
    img = np.array(img)
    size = get_size(img)
    
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

    
