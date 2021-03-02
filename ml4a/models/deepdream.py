from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/deepdream') as _importer:
    from utils import *
    from model import *
    from dream import *
    from bookmarks import *


params = DeepDreamArgs()
model = None


def setup_deepdream():
    global model
    params.model_file = downloads.download_from_gdrive(
        '1G7_wifUk8HRjFIfYZb-A6lpPV_azld06', 
        'deepdream/tensorflow_inception_graph.pb')
    model = DeepDream(params)
    return model


def run(config, img, title=None):
    if not model:
        setup_deepdream()
    return run_deepdream(model, config, img, title)


def get_layers():
    if not model:
        setup_deepdream()
    return model.get_layers()
