from speech_recognition.dataset.preprocessor.speech_commands_scanner import SpeechCommandsDirectoryProcessor
from speech_recognition.sr_config.sr_config import *


DATADIR = "../data/speech_commands_v0/"

audio_sampling_settings = prepare_audio_sampling_settings(label_count=10,
                                                          sample_rate=SAMPLE_RATE,
                                                          clip_duration_ms=CLIP_DURATION_MS,
                                                          window_size_ms=WINDOW_SIZE_MS,
                                                          window_stride_ms=WINDOW_STRIDE_MS,
                                                          dct_coefficient_count=DCT_COEFFICIENT_COUNT)

class SpeechCommandsV0(SpeechCommandsDirectoryProcessor):
    def __init__(self,
                 data_url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                 data_dir=DATADIR,
                 silence_percentage=SILENCE_PERCENTAGE,
                 unknown_percentage=UNKNOWN_PERCENTAGE,
                 possible_commands=POSSIBLE_COMMANDS,
                 validation_percentage=VALIDATION_PERCENTAGE,
                 testing_percentage=TESTING_PERCENTAGE):

        SpeechCommandsDirectoryProcessor.__init__(self, data_url=data_dir,
                                                  data_dir=data_dir,
                                                  silence_percentage=silence_percentage,
                                                  unknown_percentage=unknown_percentage,
                                                  possible_commands=possible_commands,
                                                  validation_percentage=validation_percentage,
                                                  testing_percentage=testing_percentage)