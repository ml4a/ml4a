print("import ml4a/models/submodules/__init__")
#from . import stylegan2

#from .stylegan2 import *



from . import stylegan2



import os
import sys




def submodule_in_path(submodule_name, included):
    from ml4a.models import submodules
    submodules_root = os.path.dirname(submodules.__file__)
    submodule = os.path.join(submodules_root, submodule_name)
    if included:
        sys.path.append(submodule)
    else:
        sys.path = [p for p in sys.path if p != submodule]
