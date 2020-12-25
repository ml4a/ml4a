from torchvision.models import inception_v3

from . import submodules

with submodules.localimport('submodules/torch-dreams') as _importer:
    from torch_dreams.dreamer import dreamer