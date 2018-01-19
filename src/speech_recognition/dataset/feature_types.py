import sys
sys.path.append("../")


class SingleFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

    def __eq__(self, other):
        return self.FEATURE_NAME == other.FEATURE_NAME


class MFCCFeature(SingleFeature):
    FEATURE_NAME = "mfcc_audio_data"
    FEATURE_1 = "audio_freq_spectrum"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"
