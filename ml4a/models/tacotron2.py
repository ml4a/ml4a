import numpy as np
import torch
import cv2
import matplotlib.pylab as plt
#from localimport import localimport

from .. import image
from ..utils import downloads
from ..utils import EasyDict
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/tacotron2') as _importer:
    from hparams import create_hparams
    from model import Tacotron2
    from layers import TacotronSTFT, STFT
    from audio_processing import griffin_lim
    from train import load_model
    from text import text_to_sequence
    from waveglow.denoiser import Denoiser


model = None

def setup():
    global model, waveglow, denoiser, hparams
    
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    checkpoint_path = downloads.download_from_gdrive(
        gdrive_fileid='1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA',
        output_path='tacotron2/tacotron2_statedict.pt')
    
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    
    waveglow_path = downloads.download_from_gdrive(
        gdrive_fileid='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
        output_path='tacotron2/waveglow_256channels_universal_v5.pt')

    with submodules.localimport('submodules/tacotron2/waveglow') as _importer:
        waveglow_ = torch.load(waveglow_path)
        waveglow = waveglow_['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    

def run(text, denoise=True):
    if model is None:
        setup()
    
    words = text.split(' ')
    speech_all = np.array([])
    
    excerpt_length = 20
    for w in range(0, len(words), excerpt_length):
        w1, w2 = w, min(w+excerpt_length, len(words))
        excerpt = ' '.join(words[w1:w2])

        sequence = np.array(text_to_sequence(excerpt, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

        if denoise:
            audio = denoiser(audio, strength=0.01)[:, 0]
        
        speech_all = np.concatenate([speech_all, audio.cpu().numpy()[0]], axis=-1)
    
    output = EasyDict({
        'wav': speech_all, 
        'sampling_rate': hparams.sampling_rate}
    )
    
    return output



