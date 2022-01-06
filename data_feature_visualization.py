import os
import librosa
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import scipy.io.wavfile

def visualize_data(audio_file):
    """To visualize the data in wave format"""
    data, sampling_rate = librosa.load(audio_file)
    plt.figure(figsize=(15,5))
    return librosa.display.waveplot(data, sr=sampling_rate)


def visualize_feature(audio_file):
    """To visualize extracted feaures in number format"""
    sr, x = scipy.io.wavfile.read(audio_file)
    mfcc_feat = mfcc(x, sr)
    plt.plot(mfcc_feat)
    plt.show()
    return print(mfcc_feat[0])
