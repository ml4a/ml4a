from ..utils import downloads
from . import submodules


with submodules.import_from('neural_style'):
    from utils import *
    from stylenet import *
    from generate import *


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


def run(config):
    if model is None:
        setup_neuralstyle()
    return style_transfer(model, config)