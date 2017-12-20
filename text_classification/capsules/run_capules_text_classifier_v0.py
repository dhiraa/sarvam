import os
import sys

# add utils path
sys.path.append("../")

from utils.data.kaggle.spooky_dataset import *
from capsules import capsules_text_classifier
from utils.tf_hooks.early_stopping import EarlyStoppingLossHook

#Model Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
MAX_DOC_LENGTH = 100
MAX_WORD_LENGTH = 15
NUM_CLASSES = 3 #TODO get it from dataset
DATA_STORE_PATH="tmp/"
MODEL_STORE_PATH = "tmp/bilstm_v0/"
TEXT_COL = "text"
CATEOGORY_COL = "author"


#Prepare the dataset
dataset: TextDataFrame = TextDataFrame(train_file_path=TRAIN_FILE_PATH,
                                       test_file_path=TEST_FILE_PATH,
                                       text_col=TEXT_COL,
                                       category_col=CATEOGORY_COL,
                                       model_name=DATA_STORE_PATH,
                                       max_doc_legth=MAX_DOC_LENGTH,
                                       max_word_length=MAX_WORD_LENGTH)

#To get text word ids
train_text_word_ids = dataset.get_train_text_word_ids()
val_text_word_ids = dataset.get_val_text_word_ids()
test_text_word_ids = dataset.get_test_text_word_ids()

#To get text word char IDS
train_text_word_char_ids = dataset.get_train_text_word_char_ids()
val_text_word_char_ids = dataset.get_val_text_word_char_ids()
test_text_word_char_ids = dataset.get_test_text_word_char_ids()

# To get on-hot encoded labels:
train_one_hot_encoded_label = dataset.get_train_one_hot_label()
val_one_hot_encoded_label = dataset.get_val_one_hot_label()
# dataset.get_one_hot_test_label()

#Prepare Tensorflow Dataset iterator for Estimator APIs
train_input_fn, train_input_hook = setup_input_graph2(word_ids=train_text_word_ids,
                                                      char_ids=train_text_word_char_ids,
                                                      labels=train_one_hot_encoded_label,
                                                      batch_size=BATCH_SIZE,
                                                      is_eval = False,
                                                      shuffle=True,
                                                      scope='train-data')

eval_input_fn, eval_input_hook = setup_input_graph2(word_ids=val_text_word_ids,
                                                    char_ids=val_text_word_char_ids,
                                                    labels=val_one_hot_encoded_label,
                                                    batch_size=BATCH_SIZE,
                                                    is_eval = True,
                                                    shuffle=True,
                                                    scope='val-data')

test_input_fn = test_inputs2(word_ids=test_text_word_ids,
                             char_ids=test_text_word_char_ids,
                             batch_size=1,
                             scope='test-data')

# early_stopping_hook = EarlyStoppingLossHook("reduced_mean:0", 0.030)

model = capsules_text_classifier.CapsulesTextClassifierV0(word_vocab_size=dataset.WORD_VOCAB_SIZE,
                                                          char_vocab_size=dataset.CHAR_VOCAB_SIZE,
                                                          max_doc_length=MAX_DOC_LENGTH,
                                                          max_word_length=MAX_WORD_LENGTH,
                                                          num_classes=NUM_CLASSES
                                                          )


NUM_STEPS = dataset.num_train_samples // BATCH_SIZE

#Evaluate after each epoch
for i in range(NUM_EPOCHS):
    model.train(input_fn=train_input_fn,
                hooks=[train_input_hook],
                steps=i+1 * NUM_STEPS)

    model.evaluate(input_fn=eval_input_fn, hooks=[eval_input_hook])

#Prediciting with above trained model
predictions_fn = model.predict(input_fn=test_input_fn)

predictions = []
classes = []

for r in predictions_fn:
    predictions.append(r['probabilities'])
    classes.append(r['classes'])

for i, p in enumerate(predictions_fn):
    tf.logging.info("Prediction %s: %s" % (i + 1, p["ages"]))

# Get test ids from the dataset
ids = dataset.test_df['id']
# Create a Dataframe
results = pd.DataFrame(predictions, columns=['EAP', 'HPL', 'MWS'])
results.insert(0, "id", ids)
# Store the results as expected form
results.to_csv("tmp/fast_text_v0_output.csv", index=False)
