import sys
sys.path.append("../")
import numpy as np
import pandas as pd
# from audio_utils.data.kaggle.spooky_dataset import *
from tc_utils.text_data import *
import spacy
from overrides import overrides
from tc_utils.dataset import TextClassificationDataset
from tc_utils.text_data import TextDataFrame
# nlp = spacy.load('en_core_web_sm')

DATA_STORE_PATH="jigsaw_toxic_comment_classification_challenge_data"
TEXT_COL = "comment_text"
CATEOGORY_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

TRAIN_FILE_PATH = "../data/jigsaw_toxic_comment_classification_challenge/train.csv"
TEST_FILE_PATH =  "../data/jigsaw_toxic_comment_classification_challenge/test.csv"

# #Prepare the dataset
# dataset = TextDataFrame(train_file_path=TRAIN_FILE_PATH,
#                                        test_file_path=TEST_FILE_PATH,
#                                        text_col=TEXT_COL,
#                                        category_cols=CATEOGORY_COLS,
#                                        dataset_name=DATA_STORE_PATH,
#                                        category_col = None,
#                                        max_doc_legth = 100,
#                                        max_word_length = 10,
#                                        is_multi_label=True,
#                                        is_tokenize=True)
#
#
# #To get text word ids
# train_text_word_ids = dataset.get_train_text_word_ids()
# val_text_word_ids = dataset.get_val_text_word_ids()
# # test_text_word_ids = dataset.get_test_text_word_ids()
#
# #To get text word char IDS
# train_text_word_char_ids = dataset.get_train_text_word_char_ids()
# val_text_word_char_ids = dataset.get_val_text_word_char_ids()
# # test_text_word_char_ids = dataset.get_test_text_word_char_ids()
#
# train_one_hot_encoded_label = dataset.get_train_one_hot_label()
# val_one_hot_encoded_label= dataset.get_val_one_hot_label()


class JigsawDataset(TextClassificationDataset):
    def __init__(self,
                 train_file_path,
                 test_file_path):
        TextClassificationDataset.__init__(self,
                                           train_file_path,
                                           test_file_path,
                                           "jigsaw_dataset")

        def prepare(self):
            self.dataframe = TextDataFrame(train_file_path=self.train_file_path,
                                           test_file_path=self.test_file_path,
                                           text_col="comment_text",
                                           category_col=None,
                                           category_cols=["toxic","severe_toxic","obscene","threat","insult","identity_hate"],
                                           max_doc_legth=100,
                                           max_word_length=10,
                                           is_multi_label=True,
                                           is_tokenize=True,
                                           dataset_name=self.dataset_name)
