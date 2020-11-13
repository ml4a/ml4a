import IPython
import numpy as np
import torch
import librosa
import librosa.display
import scipy.io.wavfile
    


def plot(wav, sampling_rate=None):
    wav, sampling_rate = __preprocess_wav__(wav, sampling_rate)
    assert sampling_rate is not None, 'Error: unknown sampling_rate'
    waveplot = librosa.display.waveplot(wav, sr=sampling_rate)
    return waveplot


def display(wav, sampling_rate=None):
    wav, sampling_rate = __preprocess_wav__(wav, sampling_rate)
    assert sampling_rate is not None, 'Error: unknown sampling_rate'
    display = IPython.display.Audio(wav, rate=sampling_rate)
    return display


def save(filename, wav, sampling_rate=None):
    wav, sampling_rate = __preprocess_wav__(wav, sampling_rate)
    assert sampling_rate is not None, 'Error: unknown sampling_rate'
    scipy.io.wavfile.write(
        filename, 
        sampling_rate, 
        wav)


def get_duration(wav, sampling_rate=None):
    wav, sampling_rate = __preprocess_wav__(wav, sampling_rate)
    assert sampling_rate is not None, 'Error: unknown sampling_rate'
    seconds = len(wav)/sampling_rate
    return seconds
    
    
def __preprocess_wav__(wav, sampling_rate=None):
    if isinstance(wav, str):
        wav, sampling_rate = librosa.load(wav)
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if wav.ndim > 1:
        wav = wav[0]
    return wav, sampling_rate