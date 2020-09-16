from pathlib import Path
from collections import OrderedDict
from torch.utils import model_zoo
import torch
import os
import requests
import shutil
import gdown


ML4A_DATA_ROOT = os.path.join(os.path.expanduser('~'), '.ml4a')


def get_ml4a_downloads_folder():
    global ML4A_DATA_ROOT
    ml4a_downloads = os.path.join(ML4A_DATA_ROOT, 'models')
    Path(ml4a_downloads).mkdir(parents=True, exist_ok=True)
    return ml4a_downloads

    
def download_from_gdrive(gdrive_fileid, output_path, overwrite=False):
    ml4a_downloads = get_ml4a_downloads_folder()
    folder, filename = os.path.split(output_path)
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    if not os.path.exists(output_filename) or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        gdrive_url = 'https://drive.google.com/uc?id=%s'%gdrive_fileid
        gdown.download(gdrive_url, output_filename, quiet=False)
    return output_filename


def download_data_file(url, output_path, overwrite=False):
    ml4a_downloads = get_ml4a_downloads_folder()
    folder, filename = os.path.split(output_path)
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    if not os.path.exists(output_filename) or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            with open(output_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    return output_filename


def download_neural_style(url, output_path):
    ml4a_downloads = get_ml4a_downloads_folder()
    folder, filename = os.path.split(output_path)
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    if not os.path.exists(output_filename):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        sd = model_zoo.load_url(url)
        map_ = {'classifier.1.weight':u'classifier.0.weight', 
               'classifier.1.bias':u'classifier.0.bias', 
               'classifier.4.weight':u'classifier.3.weight', 
               'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map_[k] if k in map_ else k,v) for k,v in sd.items()])
        torch.save(sd, output_filename)
    return output_filename