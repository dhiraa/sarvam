import sys

from nlp.text_classification.dataset.iterators.text_char_ids import TextAndCharIds

sys.path.append("../")

from nlp.text_classification.dataset.feature_types import TextAndCharIdsMultiLabelFeature
from sarvam.helpers.print_helper import *

class TextAndCharIdsMultiLabel(TextAndCharIds):
    def __init__(self, batch_size, dataframe, num_epochs=-1):
        TextAndCharIds.__init__(self, batch_size=batch_size, dataframe=dataframe)
        self.feature_type = TextAndCharIdsMultiLabelFeature

        try:
            if dataframe.is_multi_label == False:
                print_error("Selected dataset doesn't support multi label classification!")
                exit(0)
        except:
            pass