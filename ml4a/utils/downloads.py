from pathlib import Path
from collections import OrderedDict
from torch.utils import model_zoo
import urllib.request
import zipfile
import torch
import os
import requests
import shutil
import gdown


ML4A_DATA_ROOT = os.path.join(os.path.expanduser('~'), '.ml4a')


def get_ml4a_folder(subfolder=None):
    ml4a_folder = ML4A_DATA_ROOT
    if subfolder is not None and subfolder:
        ml4a_folder = os.path.join(ml4a_folder, subfolder)    
    Path(ml4a_folder).mkdir(parents=True, exist_ok=True)
    return ml4a_folder


def get_ml4a_downloads_folder():
    return get_ml4a_folder(None)


def get_ml4a_scratch_folder():
    return get_ml4a_folder('_scratch')


def get_ml4a_data_folder():
    return get_ml4a_folder('_data')


def unzip(zip_file, output_folder, erase_zipfile=True):
    with zipfile.ZipFile(zip_file, 'r') as zipref:
        zipref.extractall(output_folder)
        zipref.close()
    if erase_zipfile:
        os.remove(zip_file)


def download_from_gdrive(gdrive_fileid, output_path, zip_file=False, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(output_path, zip_file)
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        gdrive_url = 'https://drive.google.com/uc?id=%s'%gdrive_fileid
        gdown.download(gdrive_url, output_filename, quiet=False)
        if zip_file:
            unzip(output_filename, output_folder)
    output = output_folder if zip_file else output_filename
    return output


def download_data_file(url, output_path, zip_file=False, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(output_path, zip_file)
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print('Downloading %s to %s' % (url, output_folder))
        with requests.get(url, stream=True) as r:
            with open(output_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        if zip_file:
            unzip(output_filename, output_folder)
    output = output_folder if zip_file else output_filename
    return output

def download_text_file(url, output_path, overwrite=False):
    output_folder, output_filename, output_exists = __process_output_path__(output_path, False)
    if not output_exists or overwrite:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        filedata = urllib.request.urlopen(url)
        with open(output_filename, 'wb') as f:
            f.write(filedata.read())
    return output_filename


def download_neural_style(url, output_path):
    output_folder, output_filename, output_exists = __process_output_path__(output_path, False)
    if not output_exists:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        sd = model_zoo.load_url(url)
        map_ = {'classifier.1.weight':u'classifier.0.weight', 
               'classifier.1.bias':u'classifier.0.bias', 
               'classifier.4.weight':u'classifier.3.weight', 
               'classifier.4.bias':u'classifier.3.bias'}
        sd = OrderedDict([(map_[k] if k in map_ else k,v) for k,v in sd.items()])
        torch.save(sd, output_filename)
    return output_filename


def __process_output_path__(output_path, zip_file):
    ml4a_downloads = get_ml4a_downloads_folder()
    if zip_file:
        folder, filename = output_path, 'temp.zip'
    else:
        folder, filename = os.path.split(output_path)
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    output_exists = os.path.exists(output_folder if zip_file else output_filename)
    return output_folder, output_filename, output_exists
