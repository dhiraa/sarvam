from speech_recognition.dataset.preprocessor.simple_speech_preprocessor import SimpleSpeechPreprocessor
from speech_recognition.sr_config.sr_config import *


DATADIR = "../data/speech_commands_v0/"


class KaggleDS(SimpleSpeechPreprocessor):
    def __init__(self,
                 data_url="",
                 data_dir="../data/tensorflow_speech_recoginition_challenge/"):
        SimpleSpeechPreprocessor.__init__(self,
                 data_dir=data_dir,
                 possible_speech_commands='yes no up down left right on off stop go'.split(),
                 batch_size=16)
