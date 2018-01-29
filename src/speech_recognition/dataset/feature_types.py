import sys

from speech_recognition.sr_config.sr_config import *

sys.path.append("../")


class SingleFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

    def __eq__(self, other):
        return self.FEATURE_NAME == other.FEATURE_NAME


class RawWavAudio(SingleFeature):
    FEATURE_NAME = "raw_audio_data"
    FEATURE_1 = "raw_audio_data"
    TARGET = "command_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"

class MFCCFeature(SingleFeature):
    FEATURE_NAME = "mfcc_audio_data"
    FEATURE_1 = "audio_freq_spectrum"
    TARGET = "command_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"

    audio_sampling_settings = prepare_audio_sampling_settings(label_count=12,
                                                                    sample_rate=SAMPLE_RATE,
                                                                    clip_duration_ms=CLIP_DURATION_MS,
                                                                    window_size_ms=WINDOW_SIZE_MS,
                                                                    window_stride_ms=WINDOW_STRIDE_MS,
                                                                    dct_coefficient_count=DCT_COEFFICIENT_COUNT)
