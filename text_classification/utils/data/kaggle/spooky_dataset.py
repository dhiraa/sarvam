from utils.pandas import *
from utils.tf_data_iterators import *
from utils.spacy import *


TRAIN_FILE_PATH = "../../utils/spooky-author-identification/input/train.csv"
TEST_FILE_PATH = "../../utils/spooky-author-identification/input/test.csv"

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
train_data = dataset.get_train_data()
val_data = dataset.get_val_data()
test_data = dataset.get_test_data()

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
                                                      scope='train-utils')

eval_input_fn, eval_input_hook =  setup_input_graph(dataset.get_val_data(),
                                                     val_one_hot_encoded_label,
                                                    batch_size=BATCH_SIZE, 
                                                    scope='eval-utils')
                                                                                                          
test_input_fn =  test_inputs(dataset.get_test_data(), 
                                        batch_size=1, 
                                        scope='test-utils')
                                        

'''








