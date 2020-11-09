import os
import subprocess
import numpy as np
import PIL
import cv2
import torch
from tqdm import tqdm
from localimport import localimport

from .. import image
from ..utils import downloads
from . import submodules

#with localimport('submodules/Wav2Lip') as _importer:
with submodules.import_from('tacotron2'):  # localimport fails here   
    from hparams import create_hparams
    from model import Tacotron2
    from layers import TacotronSTFT, STFT
    from audio_processing import griffin_lim
    from train import load_model
    from text import text_to_sequence
    from waveglow.denoiser import Denoiser
    
    
# with submodules.import_from('tacotron2/waveglow'):  # localimport fails here   
#     from denoiser import Denoiser

import sys
#sys.path.append('/home/bzion/projects/ml4a/ml4a-guides/ml4a/models/submodules/tacotron2')
# from hparams import create_hparams
# from model import Tacotron2
# from layers import TacotronSTFT, STFT
# from audio_processing import griffin_lim
# from train import load_model
# from text import text_to_sequence

#sys.path.append('/home/bzion/projects/ml4a/ml4a-guides/ml4a/models/submodules/tacotron2/waveglow')
# from waveglow.denoiser import Denoiser


# with submodules.import_from('tacotron2'):  # localimport fails here   
#     from hparams import create_hparams
#     from model import Tacotron2
#     from layers import TacotronSTFT, STFT
#     from audio_processing import griffin_lim
#     from train import load_model
#     from text import text_to_sequence
    
# with submodules.import_from('tacotron2/waveglow'):  # localimport fails here   
#     from denoiser import Denoiser

    
    
#%matplotlib inline
import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd
import numpy as np
import torch

#import sys
#sys.path.append('waveglow/')


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')

model = None


def setup():
    global model, waveglow, denoiser, hparams
    
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    print("go0")
    checkpoint_path = downloads.download_from_gdrive(
        gdrive_fileid='1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA',
        output_path='tacotron2/tacotron2_statedict.pt')
    
#     checkpoint_path = 
        
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    
    print("go1")
    waveglow_path = downloads.download_from_gdrive(
        gdrive_fileid='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
        output_path='tacotron2/waveglow_256channels_universal_v5.pt')
    print("go2")
    
    
    with localimport('submodules/tacotron2/waveglow') as _importer:

        print(waveglow_path)
        waveglow2 = torch.load(waveglow_path)
        print(waveglow2)
        waveglow = waveglow2['model']
        print(waveglow)
    
    
    print("go3")
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    print("go4")
    denoiser = Denoiser(waveglow)
    print("go5")
    

def run(text):
    if model is None:
        setup()
    
    print("run1")
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    print("run2")
    
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    print("run3")
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    print("run4")
    
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    print("run5")
    
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    print("run6")
    print("run7")
    #return ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate) 
    return ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
