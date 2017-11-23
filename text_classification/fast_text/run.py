import os
import sys
#add sarvam_utils path
sys.path.append("../../")
sys.path.append(".")

from sarvam_utils.data.kaggle.spooky_dataset import *
from text_classification.fast_text import fast_text_v0
from sarvam_utils.early_stopping import EarlyStoppingLossHook

BATCH_SIZE = 32

dataset: TextDataFrame = TextDataFrame(train_df_path=TRAIN_FILE_PATH,
                        test_df_path=TEST_FILE_PATH,
                        text_col="text",
                        category_col="author",
                        model_name="fast-text-v0")

# To get the features:
train_data = dataset.get_train_data()
val_data = dataset.get_val_data()
test_data = dataset.get_test_data()

# To get indexed category labels:
train_label = dataset.get_train_label()
val_label = dataset.get_val_label()
# test_label = dataset.get_test_label()

# To get on-hot encoded labels:
train_one_hot_encoded_label = dataset.get_train_one_hot_label()
val_one_hot_encoded_label = dataset.get_val_one_hot_label()
# dataset.get_one_hot_test_label()

train_input_fn, train_input_hook = setup_input_graph(train_data,
                                                     train_one_hot_encoded_label,
                                                     batch_size=BATCH_SIZE,
                                                     scope='train-data')

eval_input_fn, eval_input_hook = setup_input_graph(dataset.get_val_data(),
                                                   val_one_hot_encoded_label,
                                                   batch_size=1,
                                                   is_eval=True,
                                                   scope='eval-data')

test_input_fn = test_inputs(dataset.get_test_data(),
                            batch_size=1,
                            scope='test-data')


config = fast_text_v0.FastTextConfig(vocab_size=dataset.vocab_count,
                                     model_dir="fast-text-v0/model/",
                                     words_vocab_file=dataset.words_vocab_file)

model = fast_text_v0.FastTextV0(config)

NUM_EPOCHS = 6
NUM_STEPS = dataset.num_train_samples // BATCH_SIZE
NUM_STEPS


early_stopping_hook = EarlyStoppingLossHook("reduced_mean:0", 0.30)


model.train(input_fn=train_input_fn, hooks=[train_input_hook, early_stopping_hook], steps=NUM_EPOCHS*NUM_STEPS)

model.evaluate(input_fn=eval_input_fn, hooks=[eval_input_hook])