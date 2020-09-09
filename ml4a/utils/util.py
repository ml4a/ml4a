import IPython
import os
from pathlib import Path
import gdown


ML4A_DATA_ROOT = os.path.join(os.path.expanduser('~'), '.ml4a')

class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

class ProgressBar:
    
    def __init__(self, total_iter, num_increments=32):
        self.num_increments = num_increments
        self.idx_iter, self.total_iter = 0, total_iter
        self.iter_per = self.total_iter / self.num_increments

    def update(self, update_str=''):
        self.idx_iter += 1
        progress_iter = int(self.idx_iter / self.iter_per)
        progress_str  = '[' + '=' * progress_iter 
        progress_str += '-' * (self.num_increments - progress_iter) + ']'
        IPython.display.clear_output(wait=True)
        IPython.display.display(progress_str+'  '+update_str)


def log(message, verbose=True):
    if not verbose:
        return
    print(message)


def warn(condition, message, verbose=True):
    if not condition:
        return
    log('Warning: %s' % message, verbose)


    
    
###########









def get_ml4a_downloads_folder():
    global ML4A_DATA_ROOT
    ml4a_downloads = os.path.join(ML4A_DATA_ROOT, 'models')
    Path(ml4a_downloads).mkdir(parents=True, exist_ok=True)
    return ml4a_downloads
    
def download_from_gdrive(gdrive_fileid, output_path):
    folder, filename = os.path.split(output_path)
    ml4a_downloads = get_ml4a_downloads_folder()
    output_folder = os.path.join(ml4a_downloads, folder)
    output_filename = os.path.join(output_folder, filename)
    if not os.path.exists(output_filename):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        gdrive_url = 'https://drive.google.com/uc?id=%s'%gdrive_fileid
        gdown.download(gdrive_url, output_filename, quiet=False)
    return output_filename
