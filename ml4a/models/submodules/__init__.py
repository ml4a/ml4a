import os
import sys
import numba.cuda
from .. import submodules
from .localimport import localimport



def cuda_available():
    available = numba.cuda.is_available()
    if not available:
        print('Warning! No GPU detected, so most models will fail... If you are in Colab, make sure you enable GPU in the menu: Runtime -> Change runtime type.')
    return available


def get_submodules_root(submodule_name):
    submodules_root = os.path.dirname(submodules.__file__)
    return os.path.join(submodules_root, submodule_name)
    
    
class import_from(object):
    
    def __init__(self, submodule_name):
        self.submodule = get_submodules_root(submodule_name)
        sys.path.append(self.submodule)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        sys.path = [path for path in sys.path if path != self.submodule]
        return False
