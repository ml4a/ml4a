from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/neural_style') as _importer:
    from utils import *
    from model import *
    from stylenet import *


params = StylenetArgs()
model = None


def setup_neuralstyle():
    global model
    params.model_file = downloads.download_neural_style(
        'https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth', 
        'neural_style/vgg19-d01eb7cb.pth')
    dtype, multidevice, backward_device = setup_gpu(params)
    model = StyleNet(params, dtype, multidevice, backward_device, verbose=False)
    return model


def run(config, img=None, title=None):
    if model is None:
        setup_neuralstyle()
    return style_transfer(model, config, img, title)
