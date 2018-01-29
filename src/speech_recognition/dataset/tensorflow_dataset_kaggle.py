from speech_recognition.dataset.preprocessor.simple_speech_preprocessor import SimpleSpeechPreprocessor
from speech_recognition.sr_config.sr_config import *


DATADIR = "../data/speech_commands_v0/"

audio_sampling_settings = prepare_audio_sampling_settings(label_count=10,
                                                          sample_rate=SAMPLE_RATE,
                                                          clip_duration_ms=CLIP_DURATION_MS,
                                                          window_size_ms=WINDOW_SIZE_MS,
                                                          window_stride_ms=WINDOW_STRIDE_MS,
                                                          dct_coefficient_count=DCT_COEFFICIENT_COUNT)


class SpeechCommandsV0(SimpleSpeechPreprocessor):
    def __init__(self,
                 data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                 data_dir="../data/tensorflow_speech_recoginition_challenge/"):
        SimpleSpeechPreprocessor.__init__(self,
                 data_dir=data_dir,
                 possible_speech_commands='yes no up down left right on off stop go _silence_ unknown'.split(),
                 batch_size=16)
        self.audio_sampling_settings = audio_sampling_settings
