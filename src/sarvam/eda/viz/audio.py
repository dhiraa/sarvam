# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
from sarvam.helpers.print_helper import *

def get_spectrum(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def log_specgram(wave_file_path, window_size=20,
                 step_size=10, eps=1e-10):
    '''
    
    Note, that we are taking logarithm of spectrogram values. 
    It will make our plot much more clear, moreover, 
    it is strictly connected to the way people hear. 
    We need to assure that there are no 0 values as input to logarithm.
    :param audio: 
    :param sample_rate: 
    :param window_size: 
    :param step_size: 
    :param eps: 
    :return: 
    '''

    sample_rate, audio = wavfile.read(wave_file_path)

    freqs, times, spectrogram = get_spectrum(audio=audio, sample_rate=sample_rate)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + wave_file_path)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate / len(audio), sample_rate), audio)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title('Spectrogram of ' + wave_file_path)
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')

    return None


def melspectrogram(wave_file_path, is_delta=False):

    sample_rate, audio = wavfile.read(wave_file_path)

    print_info("audio.shape: " + str(audio.shape))

    S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=64)
    log_S = librosa.power_to_db(S, ref=np.max)

    print_info("log_S.shape: " + str(log_S.shape))

    if is_delta:
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        # Let's pad on the first and second deltas while we're at it
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        print_info("delta2_mfcc.shape: " + str(delta2_mfcc.shape))

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(delta2_mfcc)
        plt.ylabel('MFCC coeffs')
        plt.xlabel('Time')
        plt.title('MFCC')
        plt.colorbar()
        plt.tight_layout()
    else:
        # Convert to log scale (dB). We'll use the peak power (max) as reference.

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.title('Mel power spectrogram ')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()


def specgram_3d(wave_file_path, window_size=20,
                 step_size=10, eps=1e-10):
    '''

    Note, that we are taking logarithm of spectrogram values. 
    It will make our plot much more clear, moreover, 
    it is strictly connected to the way people hear. 
    We need to assure that there are no 0 values as input to logarithm.
    :param audio: 
    :param sample_rate: 
    :param window_size: 
    :param step_size: 
    :param eps: 
    :return: 
    '''

    sample_rate, audio = wavfile.read(wave_file_path)

    freqs, times, spectrogram = get_spectrum(audio=audio, sample_rate=sample_rate)

    data = [go.Surface(z=spectrogram.T)]
    layout = go.Layout(
        title='Specgtrogram of "yes" in 3d',
        scene=dict(
            yaxis=dict(title='Frequencies', range=freqs),
            xaxis=dict(title='Time', range=times),
            zaxis=dict(title='Log amplitude'),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)