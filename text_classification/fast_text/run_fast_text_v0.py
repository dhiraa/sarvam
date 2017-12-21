import sys
sys.path.append("../")

# add tc_utils path
from tc_utils.tf_data_iterators import *
from kaggle.dataset.spooky import *
from fast_text import fast_text_v0
from tc_utils.tf_hooks.early_stopping import EarlyStoppingLossHook

#Model Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 6
DATA_STORE_PATH="tmp/"
MODEL_STORE_PATH = "tmp/fast_text_v0/"
TEXT_COL = "text"
CATEOGORY_COL = "author"

#Prepare the dataset
dataset = TextDataFrame(train_file_path=TRAIN_FILE_PATH,
                                       test_file_path=TEST_FILE_PATH,
                                       text_col=TEXT_COL,
                                       category_col=CATEOGORY_COL,
                                       dataset_name=DATA_STORE_PATH)

train_data = dataset.get_train_text_data()
val_data = dataset.get_val_text_data()
test_data = dataset.get_test_text_data()

# To get indexed category labels:
train_label = dataset.get_train_label()
val_label = dataset.get_val_label()
# test_label = dataset.get_test_label()

# To get on-hot encoded labels:
train_one_hot_encoded_label = dataset.get_train_one_hot_label()
val_one_hot_encoded_label = dataset.get_val_one_hot_label()
# dataset.get_one_hot_test_label()

#Prepare Tensorflow Dataset iterator for Estimator APIs
train_input_fn, train_input_hook = setup_input_graph(train_data,
                                                     train_one_hot_encoded_label,
                                                     batch_size=BATCH_SIZE,
                                                     scope='train-audio_utils')

eval_input_fn, eval_input_hook = setup_input_graph(val_data,
                                                   val_one_hot_encoded_label,
                                                   batch_size=1,
                                                   is_eval=True,
                                                   scope='eval-audio_utils')

test_input_fn = test_inputs(test_data,
                            batch_size=1,
                            scope='test-audio_utils')

# Configure the model
config = fast_text_v0.FastTextConfig(vocab_size=dataset.vocab_count,
                                     model_dir=MODEL_STORE_PATH,
                                     words_vocab_file=dataset.words_vocab_file)

early_stopping_hook = EarlyStoppingLossHook("reduced_mean:0", 0.030)

model = fast_text_v0.FastTextV0(config)


NUM_STEPS = dataset.num_train_samples // BATCH_SIZE

#Evaluate after each epoch
for i in range(NUM_EPOCHS):
    model.train(input_fn=train_input_fn,
                hooks=[train_input_hook, early_stopping_hook],
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
