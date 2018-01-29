import os

import pandas as pd

from nlp.text_classification.dataset.dataframe import TextDataFrame
from nlp.text_classification.dataset.dataset import TextClassificationDataset

TRAIN_FILE_PATH = "../data/spooky_author_identification/input/train.csv"
TEST_FILE_PATH = "../data/spooky_author_identification/input/test.csv"

TEXT_COL = "text"
CATEGORY_COL = "author"
PROCESSED_COL = "spacy_processed"

'''
Usage:
import sys
sys.path.append("path/to/sarvam/")

BATCH_SIZE = 32

dataset = TextDataFrame(train_df_path=TRAIN_FILE_PATH,
                        test_df_path=TEST_FILE_PATH,
                        text_col="text",
                        category_col="author",
                        model_name="fast-text-v0-")
                        
# To get the features:
train_data = dataset.get_train_text_data()
val_data = dataset.get_val_text_data()
test_data = dataset.get_test_text_data()

# To get indexed category labels:
train_label = dataset.get_train_label()
val_label = dataset.get_val_label()
# test_label = dataset.get_test_label()

#To get on-hot encoded labels:
train_one_hot_encoded_label = dataset.get_train_one_hot_label()
val_one_hot_encoded_label= dataset.get_val_one_hot_label()
# dataset.get_one_hot_test_label()

#To get Tensorflow input graph function and init hook
train_input_fn, train_input_hook = setup_input_graph(train_data,
                                                     train_one_hot_encoded_label,
                                                      batch_size=BATCH_SIZE, 
                                                      scope='train-audio_utils')

eval_input_fn, eval_input_hook =  setup_input_graph(dataset.get_val_text_data(),
                                                     val_one_hot_encoded_label,
                                                    batch_size=BATCH_SIZE, 
                                                    scope='eval-audio_utils')
                                                                                                          
test_input_fn =  test_inputs(dataset.get_test_text_data(), 
                                        batch_size=1, 
                                        scope='test-audio_utils')
                                        

'''

class SpookyDataset(TextClassificationDataset):
    def __init__(self,
                 train_file_path=TRAIN_FILE_PATH,
                 test_file_path=TEST_FILE_PATH):
        TextClassificationDataset.__init__(self,
                                           train_file_path,
                                           test_file_path,
                                           "spooky_dataset")

    def prepare(self):
        self.dataframe = TextDataFrame(train_file_path=self.train_file_path,
                                       test_file_path=self.test_file_path,
                                       text_col="text",
                                       category_col="author",
                                       category_cols=None,
                                       max_doc_legth=100,
                                       max_word_length=10,
                                       is_multi_label=False,
                                       is_tokenize=False,
                                       dataset_name=self.dataset_name)


    def predict_on_csv_files(self, data_iterator, estimator):

        predictions_fn = estimator.predict(input_fn=data_iterator.get_test_input_function())

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
        results = pd.DataFrame(predictions, columns=['EAP', 'HPL', 'MWS'])
        results.insert(0, "id", ids)
        # Store the results as expected form

        out_dir = estimator.model_dir + "/predictions/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        results.to_csv(out_dir + "/test_out.csv", index=False)










