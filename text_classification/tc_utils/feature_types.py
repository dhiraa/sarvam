import sys
sys.path.append("../")


class TwoFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    FEATURE_2 = None

class TextAndCharIdsFeature(TwoFeature):
    FEATURE_NAME = "text+char ids"
    FEATURE_1 = "text_ids"
    FEATURE_2 = "char_ids"