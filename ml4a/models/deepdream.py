from ..utils import downloads
from . import submodules


with submodules.import_from('deepdream'):
    from util import *
    from deepdream import *
    from dream import *


params = DeepDreamArgs()
model = None


def setup_deepdream():
    global model
    params.model_file = downloads.download_from_gdrive(
        '1G7_wifUk8HRjFIfYZb-A6lpPV_azld06', 
        'deepdream/tensorflow_inception_graph.pb')
    model = DeepDream(params)
    return model


def run(config, img0):
    if not model:
        setup_deepdream()
    return run_deepdream(model, config, img0)
