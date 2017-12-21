import sys
sys.path.append("../")

# add  path
from tc_utils.tf_data_iterators import setup_input_graph2



# from utils.kaggle.spooky_dataset import *
from tc_utils.kaggle.jigsaw_dataset import *
from bilstm import bilstm_v0

#Model Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
# DATA_STORE_PATH="tmp/"
MODEL_STORE_PATH = "model/bilstm_v0/"
# TEXT_COL = "text"
# CATEOGORY_COL = "author"
#
#
# #Prepare the dataset
# dataset: TextDataFrame = TextDataFrame(train_file_path=TRAIN_FILE_PATH,
#                                        test_file_path=TEST_FILE_PATH,
#                                        text_col=TEXT_COL,
#                                        category_col=CATEOGORY_COL,
#                                        dataset_name=DATA_STORE_PATH)
#
# #To get text word ids
# train_text_word_ids = dataset.get_train_text_word_ids()
# val_text_word_ids = dataset.get_val_text_word_ids()
# test_text_word_ids = dataset.get_test_text_word_ids()
#
# #To get text word char IDS
# train_text_word_char_ids = dataset.get_train_text_word_char_ids()
# val_text_word_char_ids = dataset.get_val_text_word_char_ids()
# test_text_word_char_ids = dataset.get_test_text_word_char_ids()
#
# # To get on-hot encoded labels:
# train_one_hot_encoded_label = dataset.get_train_one_hot_label()
# val_one_hot_encoded_label = dataset.get_val_one_hot_label()
# # dataset.get_one_hot_test_label()

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

# test_input_fn = test_inputs2(word_ids=test_text_word_ids,
#                              char_ids=test_text_word_char_ids,
#                              batch_size=1,
#                              scope='test-data')

# Configure the model
config = bilstm_v0.BiLSTMConfig(model_dir=MODEL_STORE_PATH,
                                vocab_size=dataset.WORD_VOCAB_SIZE,
                                char_vocab_size=dataset.CHAR_VOCAB_SIZE,
                                num_classes=6,
                                #hyper parameters
                                use_char_embedding=False,
                                learning_rate=0.001,
                                word_level_lstm_hidden_size=128,
                                word_emd_size=128,
                                num_lstm_layers=2,
                                char_level_lstm_hidden_size=64,
                                char_emd_size=128,
                                out_keep_propability=0.5)

# early_stopping_hook = EarlyStoppingLossHook("reduced_mean:0", 0.030)

model = bilstm_v0.BiLSTMV0(config)


NUM_STEPS = dataset.num_train_samples // BATCH_SIZE

#Evaluate after each epoch
for i in range(NUM_EPOCHS):
    model.train(input_fn=train_input_fn,
                hooks=[train_input_hook],
                steps=i+1 * NUM_STEPS)

    model.evaluate(input_fn=eval_input_fn, hooks=[eval_input_hook])

# #Prediciting with above trained model
# predictions_fn = model.predict(input_fn=test_input_fn)
#
# predictions = []
# classes = []
#
# for r in predictions_fn:
#     predictions.append(r['probabilities'])
#     classes.append(r['classes'])
#
# for i, p in enumerate(predictions_fn):
#     tf.logging.info("Prediction %s: %s" % (i + 1, p["ages"]))
#
# # Get test ids from the dataset
# ids = dataset.test_df['id']
# # Create a Dataframe
# results = pd.DataFrame(predictions, columns=['EAP', 'HPL', 'MWS'])
# results.insert(0, "id", ids)
# # Store the results as expected form
# results.to_csv("tmp/fast_text_v0_output.csv", index=False)
