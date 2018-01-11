import sys
sys.path.append("../")
import numpy as np
import pandas as pd
# from audio_utils.data.kaggle.spooky_dataset import *
from tc_utils.dataframe import *
import spacy
from overrides import overrides
from tc_utils.dataset import TextClassificationDataset
from tc_utils.dataframe import TextDataFrame
# nlp = spacy.load('en_core_web_sm')

DATA_STORE_PATH="jigsaw_toxic_comment_classification_challenge_data"
TEXT_COL = "token"
CATEOGORY_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

TRAIN_FILE_PATH = "../data/jigsaw_toxic_comment_classification_challenge/train_preprocessed.json"
TEST_FILE_PATH =  "../data/jigsaw_toxic_comment_classification_challenge/test_preprocessed.json"


class JigsawDataset(TextClassificationDataset):
    def __init__(self,
                 train_file_path=TRAIN_FILE_PATH,
                 test_file_path=TEST_FILE_PATH):
        TextClassificationDataset.__init__(self,
                                           train_file_path,
                                           test_file_path,
                                           "jigsaw_dataset")

    def prepare(self):
        self.dataframe = TextDataFrame(train_file_path=self.train_file_path,
                                       test_file_path=self.test_file_path,
                                       text_col=TEXT_COL,
                                       category_col=None,
                                       category_cols=["toxic","severe_toxic","obscene","threat","insult","identity_hate"],
                                       max_doc_legth=150,
                                       max_word_length=10,
                                       is_multi_label=True,
                                       is_tokenize=False,
                                       dataset_name=self.dataset_name)

    def predict_on_csv_files(self, data_iterator, estimator):

        predictions_fn = estimator.predict(input_fn=data_iterator.get_test_function(), hooks=[data_iterator.get_test_hook()])

        predictions = []
        classes = []

        for r in predictions_fn:
            predictions.append(r[ estimator.feature_type.OUT_PROBABILITIES])
            classes.append(r[ estimator.feature_type.OUT_CLASSES])

        # for i, p in enumerate(predictions_fn):
        #     tf.logging.info("Prediction %s: %s" % (i + 1, p["ages"]))

        # Get test ids from the dataset
        ids = self.dataframe.test_df['id']
        # Create a Dataframe
        results = pd.DataFrame(predictions, columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"])
        results.insert(0, "id", ids)
        # Store the results as expected form

        out_file = estimator.model_dir + "/predictions/test_output.csv"
        results.to_csv(out_file, index=False)