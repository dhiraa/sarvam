import sys
sys.path.append("../")


class SingleFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

class TextFeature(SingleFeature):
    FEATURE_NAME = "text ids"
    FEATURE_1 = "text_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"

class TwoFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    FEATURE_2 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

class TextAndCharIdsFeature(TwoFeature):
    FEATURE_NAME = "text+char ids"
    FEATURE_1 = "text_ids"
    FEATURE_2 = "char_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"