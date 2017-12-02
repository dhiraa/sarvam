import os
import sys

# add utils path
sys.path.append("../../")
sys.path.append(".")

from utils.data.kaggle.spooky_dataset import *
from multi_class_cnn_rnn import multi_class_cnn_rnn_v0
from utils.tf_hooks.early_stopping import EarlyStoppingLossHook

BATCH_SIZE = 16

dataset: TextDataFrame = TextDataFrame(train_df_path=TRAIN_FILE_PATH,
                                       test_df_path=TEST_FILE_PATH,
                                       text_col="text",
                                       category_col="author",
                                       model_name="multi-class-cnn-rnn-v0")

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
                                                     scope='train-utils')

eval_input_fn, eval_input_hook = setup_input_graph(dataset.get_val_data(),
                                                   val_one_hot_encoded_label,
                                                   batch_size=1,
                                                   is_eval=True,
                                                   scope='eval-utils')

test_input_fn = test_inputs(dataset.get_test_data(),
                            batch_size=1,
                            scope='test-utils')

config = multi_class_cnn_rnn_v0.MultiClassCNNRNNConfig(vocab_size=dataset.vocab_count,
                                                       model_dir="multi-class-cnn-rnn-v0/model/",
                                                       words_vocab_file=dataset.words_vocab_file)

model = multi_class_cnn_rnn_v0.MultiClassCNNRNNV0(config)

NUM_EPOCHS = 6
NUM_STEPS = dataset.num_train_samples // BATCH_SIZE
NUM_STEPS

early_stopping_hook = EarlyStoppingLossHook("reduced_mean:0", 0.030)

model.train(input_fn=train_input_fn,
            hooks=[train_input_hook, early_stopping_hook],
            steps=NUM_EPOCHS * NUM_STEPS)

model.evaluate(input_fn=eval_input_fn, hooks=[eval_input_hook])

predictions_fn = model.predict(input_fn=test_input_fn)

predictions = []
classes = []

for r in predictions_fn:
    predictions.append(r['probabilities'])
    classes.append(r['classes'])

for i, p in enumerate(predictions_fn):
    tf.logging.info("Prediction %s: %s" % (i + 1, p["ages"]))

ids = dataset.test_df['id']
results = pd.DataFrame(predictions, columns=['EAP', 'HPL', 'MWS'])
results.insert(0, "id", ids)

results.to_csv("mccr_text_tokenized.csv", index=False)
