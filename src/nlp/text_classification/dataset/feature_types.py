import sys
sys.path.append("../")


class SingleFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

    def __eq__(self, other):
        return self.FEATURE_NAME == other.FEATURE_NAME

class TwoFeature():
    FEATURE_NAME = None
    FEATURE_1 = None
    FEATURE_2 = None
    OUT_PROBABILITIES = None
    OUT_CLASSES = None

    def __eq__(self, other):
        return (self.FEATURE_NAME == other.FEATURE_NAME &
                self.OUT_CLASSES == self.OUT_CLASSES)

class TextIdsFeature(SingleFeature):
    FEATURE_NAME = "text ids"
    FEATURE_1 = "text_ids"
    LABEL = "multiclass"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"

class TextAndCharIdsFeature(TwoFeature):
    FEATURE_NAME = "text+char ids"
    FEATURE_1 = "text_ids"
    FEATURE_2 = "char_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "classes"

class TextAndCharIdsMultiLabelFeature(TwoFeature):
    FEATURE_NAME = "text+char ids"
    FEATURE_1 = "text_ids"
    FEATURE_2 = "char_ids"
    OUT_PROBABILITIES = "probabilities"
    OUT_CLASSES = "multilabel"