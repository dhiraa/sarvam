import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorFactory():

    iterator_path = {
        "raw_audio_data": "speech_recognition.dataset.iterators.raw_audio_data",
        "audio_mfcc_google": "speech_recognition.dataset.iterators.audio_mfcc_google",
        "audio_mfcc_librosa": "speech_recognition.dataset.iterators.audio_mfcc_librosa",
    }

    iterators = {
        "raw_audio_data": "AudioMFCC",
        "audio_mfcc_google": "AudioMFCC",
        "audio_mfcc_librosa": "AudioMFCC",
    }


    def __init__(self):
        pass

    @staticmethod
    def _get_iterator(name):
        '''
        '''
        try:
            data_iterator = getattr(import_module(DataIteratorFactory.iterator_path[name]), DataIteratorFactory.iterators[name])
        except KeyError:
            raise NotImplemented("Given data iterator file name not found: {}".format(name))
        # Return the model class
        return data_iterator

    @staticmethod
    def get(iterator_name):
        iterator = DataIteratorFactory._get_iterator(iterator_name)
        return iterator


