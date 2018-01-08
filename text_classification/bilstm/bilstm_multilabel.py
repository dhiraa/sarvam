import os

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tc_utils.feature_types import TextAndCharIdsFeature
from tc_utils.tc_config import *
import pickle

tf.logging.set_verbosity("INFO")


def get_sequence_length(sequence_ids, pad_word_id=0):
    '''
    Returns the sequence length, droping out all the padded tokens if the sequence is padded

    :param sequence_ids: Tensor(shape=[batch_size, doc_length])
    :param pad_word_id: 0 is default
    :return: Array of Document lengths of size batch_size
    '''
    flag = tf.greater_equal(sequence_ids, 1) # TODO 1 -> start of <UNK> vocab id
    used = tf.cast(flag, tf.int32)
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

class BiLSTMMultiLabelConfig():
    def __init__(self,
                 model_dir,
                 vocab_size,
                 char_vocab_size,
                 num_classes,
                 max_document_length,
                 #hyper parameters
                 use_char_embedding,
                 learning_rate,
                 word_level_lstm_hidden_size,
                 char_level_lstm_hidden_size,
                 word_emd_size,
                 char_emd_size,
                 num_lstm_layers,
                 out_keep_propability):

        # tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        # tf.app.flags._global_parser = argparse.ArgumentParser()
        # flags = tf.app.flags
        # self.FLAGS = flags.FLAGS

        # Constant params
        self.MODEL_DIR = model_dir
        self.UNKNOWN_TAG = "O"

        # Preprocessing Paramaters
        self.VOCAB_SIZE = int(vocab_size)
        self.CHAR_VOCAB_SIZE = int(char_vocab_size)
        self.NUM_CLASSES = int(num_classes)

        self.MAX_DOC_LENGTH = int(max_document_length)


        # Model hyper parameters
        self.USE_CHAR_EMBEDDING = use_char_embedding
        self.LEARNING_RATE =  float(learning_rate)
        self.KEEP_PROP = float(out_keep_propability)
        self.WORD_EMBEDDING_SIZE = int(word_emd_size)
        self.CHAR_EMBEDDING_SIZE = int(char_emd_size)
        self.WORD_LEVEL_LSTM_HIDDEN_SIZE =  int(word_level_lstm_hidden_size)
        self.CHAR_LEVEL_LSTM_HIDDEN_SIZE =  int(char_level_lstm_hidden_size)
        self.NUM_LSTM_LAYERS =  int(num_lstm_layers)

    @staticmethod
    def dump(model_dir, config):
        with open(model_dir+"/model_config.pickle", "wb") as file:
            pickle.dump(config, file)

    @staticmethod
    def load(model_dir):
        with open(model_dir + "/model_config.pickle", "rb") as file:
            cfg = pickle.load(file)
        return cfg

    @staticmethod
    def user_config(dataframe):

        vocab_size = dataframe.WORD_VOCAB_SIZE
        char_vocab_size = dataframe.CHAR_VOCAB_SIZE
        num_classes = dataframe.NUM_CLASSES
        max_document_length =dataframe.MAX_DOC_LEGTH
        # hyper parameters
        use_char_embedding = input("use_char_embedding (y/n):") or "y"
        learning_rate = input("learning_rate: (0.001): ") or 0.001
        word_level_lstm_hidden_size  = input("word_level_lstm_hidden_size: (32): ") or 32
        char_level_lstm_hidden_size = input("char_level_lstm_hidden_size (16): ") or 16
        word_emd_size = input("word_emd_size (32): ") or 32
        char_emd_size= input("char_emd_size (16): ") or 16
        num_lstm_layers = input("num_lstm_layers (2): ") or 2
        out_keep_propability = input("out_keep_propability (0.5): ") or 0.5

        model_dir =  "/uce_{}_lr_{}_wlstm_{}_num_layers_{}_clstm_{}_wembd_{}_cembd_{}_keep_{}".format(
            str(use_char_embedding),
            str(learning_rate),
            str(word_level_lstm_hidden_size),
            str(num_lstm_layers),
            str(char_level_lstm_hidden_size),
            str(word_emd_size),
            str(char_emd_size),
            str(out_keep_propability)
        )

        model_dir = EXPERIMENT_MODEL_ROOT_DIR + dataframe.dataset_name + "/BiLSTMMultiLabelClassifier/" + model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        cfg =  BiLSTMMultiLabelConfig(model_dir,
                                      vocab_size,
                                      char_vocab_size,
                                      num_classes,
                                      max_document_length,
                                      #hyper parameters
                                      use_char_embedding,
                                      learning_rate,
                                      word_level_lstm_hidden_size,
                                      char_level_lstm_hidden_size,
                                      word_emd_size,
                                      char_emd_size,
                                      num_lstm_layers,
                                      out_keep_propability)

        BiLSTMMultiLabelConfig.dump(model_dir, cfg)

        return cfg

# =======================================================================================================================


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
run_config.allow_soft_placement = True
run_config.log_device_placement = False
run_config=tf.contrib.learn.RunConfig(session_config=run_config)

class BiLSTMMultiLabelClassifier(tf.estimator.Estimator):
    def __init__(self,
                 bilstm_config: BiLSTMMultiLabelConfig):
        super(BiLSTMMultiLabelClassifier, self).__init__(
            model_fn=self._model_fn,
            model_dir=bilstm_config.MODEL_DIR,
            config=run_config)

        self.bilstm_config = bilstm_config

        self.hooks = []

        self.feature_type = TextAndCharIdsFeature

    def _model_fn(self, features, labels, mode, params):
        '''

        :param features: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param labels: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param mode: ModeKeys
        :param params:
        :return:
        '''

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        token_ids = features[self.feature_type.FEATURE_1]

        # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
        char_ids = features[self.feature_type.FEATURE_2]

        print("\n\n\n\n\n")
        tf.logging.info('token_ids: =======> {}'.format(token_ids))
        tf.logging.info('char_ids: =======> {}'.format(char_ids))
        tf.logging.info('labels: =======> {}'.format(labels))

        s = tf.shape(char_ids)

        #remove pad words
        char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2])) #20 -> char dim

        with tf.name_scope("word-embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=self.bilstm_config.VOCAB_SIZE,
                                                               embed_dim=self.bilstm_config.WORD_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=self.bilstm_config.KEEP_PROP,
                                                seed=42,
                                                training=mode == tf.estimator.ModeKeys.TRAIN)

            # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
            tf.logging.info('word_embeddings =====> {}'.format(word_embeddings))

            # seq_length = get_sequence_length_old(word_embeddings) TODO working
            #[BATCH_SIZE, ]
            seq_length = get_sequence_length(token_ids)

            tf.logging.info('seq_length =====> {}'.format(seq_length))

        with tf.variable_scope("char_embed_layer"):
            char_embeddings = tf.contrib.layers.embed_sequence(char_ids,
                                                               vocab_size=self.bilstm_config.CHAR_VOCAB_SIZE,
                                                               embed_dim=self.bilstm_config.CHAR_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            #[BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.layers.dropout(char_embeddings,
                                                rate=self.bilstm_config.KEEP_PROP,
                                                seed=42,
                                                training=mode == tf.estimator.ModeKeys.TRAIN)  # TODO add test case


            tf.logging.info('char_embeddings =====> {}'.format(char_embeddings))

        with tf.variable_scope("chars_level_bilstm_layer"):
            # put the time dimension on axis=1
            shape = tf.shape(char_embeddings)

            BATCH_SIZE = shape[0]
            MAX_DOC_LENGTH = shape[1]
            CHAR_MAX_LENGTH = shape[2]

            TOTAL_DOCS_LENGTH = tf.reduce_sum(seq_length)

            # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]  ===>
            #      [BATCH_SIZE * MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.reshape(char_embeddings,
                                         shape=[BATCH_SIZE * MAX_DOC_LENGTH, CHAR_MAX_LENGTH,
                                                self.bilstm_config.CHAR_EMBEDDING_SIZE],
                                         name="reduce_dimension_1")

            tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

            # word_lengths = get_sequence_length_old(char_embeddings) TODO working
            word_lengths = get_sequence_length(char_ids_reshaped)

            tf.logging.info('word_lengths =====> {}'.format(word_lengths))

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.bilstm_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.bilstm_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                              state_is_tuple=True)

            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=word_lengths,
                inputs=char_embeddings,
                scope="encode_words")

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            encoded_words = tf.concat([output_fw, output_bw], axis=-1)

            # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
            encoded_words = tf.reshape(encoded_words,
                                       shape=[BATCH_SIZE, MAX_DOC_LENGTH, 2 *
                                              self.bilstm_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE])

            tf.logging.info('encoded_words =====> {}'.format(encoded_words))

        with  tf.name_scope("word_level_lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(self.bilstm_config.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(self.bilstm_config.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=self.bilstm_config.KEEP_PROP)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=self.bilstm_config.KEEP_PROP)
            else:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  self.bilstm_config.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  self.bilstm_config.NUM_LSTM_LAYERS,
                                                            state_is_tuple=True)

            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype=tf.float32,
                sequence_length=seq_length,
                inputs=word_embeddings,
                scope="encod_sentence")

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE) TODO check MAX_SEQ_LENGTH?
            encoded_sentence = tf.concat([fw_output_one,
                                          bw_output_one], axis=-1)

            tf.logging.info('encoded_sentence =====> {}'.format(encoded_sentence))

        with tf.variable_scope("char_word_embeddings-mergeing_layer"):

            encoded_doc = tf.concat([encoded_words, encoded_sentence], axis=-1)

            encoded_doc = tf.layers.dropout(encoded_doc, rate=self.bilstm_config.KEEP_PROP, seed=42,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

        with tf.name_scope("hidden_layer"):
            encoded_doc_shape = tf.shape(encoded_doc)
            BATCH_SIZE = encoded_doc_shape[0]

            size = ((self.bilstm_config.WORD_LEVEL_LSTM_HIDDEN_SIZE * 2) + \
                    (self.bilstm_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE * 2)) * \
                   self.bilstm_config.MAX_DOC_LENGTH

            encoded_doc = tf.reshape(encoded_doc, shape=[BATCH_SIZE, size])

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

            combined_logits = tf.layers.dense(inputs=encoded_doc,
                                              units=self.bilstm_config.NUM_CLASSES*10,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
                                              activation=tf.nn.relu)

            combined_logits = tf.layers.dense(inputs=combined_logits,
                                              units=self.bilstm_config.NUM_CLASSES,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('combined_logits: =====> {}'.format(combined_logits))

            size = (self.bilstm_config.CHAR_LEVEL_LSTM_HIDDEN_SIZE * 2) * \
                   self.bilstm_config.MAX_DOC_LENGTH

            encoded_words_hidden_layer = tf.reshape(encoded_words, shape=[BATCH_SIZE, size])

            tf.logging.info('encoded_words_hidden_layer: =====> {}'.format(encoded_words_hidden_layer))

            encoded_words_hidden_layer = tf.layers.dense(inputs=encoded_words_hidden_layer,
                                                         units=self.bilstm_config.NUM_CLASSES*10,
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
                                                         activation=tf.nn.relu)

            encoded_words_hidden_layer = tf.layers.dense(inputs=encoded_words_hidden_layer,
                                                         units=self.bilstm_config.NUM_CLASSES,
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('encoded_words_hidden_layer: =====> {}'.format(encoded_words_hidden_layer))

            size = (self.bilstm_config.WORD_LEVEL_LSTM_HIDDEN_SIZE * 2) * \
                   self.bilstm_config.MAX_DOC_LENGTH

            encoded_sentence_hidden_layer = tf.reshape(encoded_sentence, shape=[BATCH_SIZE, size])

            tf.logging.info('encoded_sentence_hidden_layer: =====> {}'.format(encoded_sentence_hidden_layer))

            encoded_sentence_hidden_layer = tf.layers.dense(inputs=encoded_sentence_hidden_layer,
                                                            units=self.bilstm_config.NUM_CLASSES*10,
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
                                                            activation=tf.nn.relu)

            encoded_sentence_hidden_layer = tf.layers.dense(inputs=encoded_sentence_hidden_layer,
                                                            units=self.bilstm_config.NUM_CLASSES,
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('encoded_sentence_hidden_layer: =====> {}'.format(encoded_sentence_hidden_layer))

        with tf.name_scope("ensemble-layer"):
            #[[class1_m1, class2_m1, class3_m1], [class1_m2, class2_m2, class3_m2], [class1_m3, class2_m3, class3_m3]]
            #[
            # [class1_m1, class1_m2, class1_m3]
            # [class2_m1, class2_m2, class2_m3]
            # [class3_m1, class3_m2, class3_m3]
            # ]
            ensemble_layer = tf.stack([combined_logits, encoded_words_hidden_layer, encoded_sentence_hidden_layer],
                                      axis=1)
            logits = ensemble_layer

            tf.logging.info('logits: =====> {}'.format(logits))

        with  tf.name_scope("loss-layer"):
            """Defines the loss"""

            if mode == ModeKeys.INFER:
                labels = tf.placeholder(tf.int32, shape=[None, None],
                                        name="labels")  # no labels during prediction
            else:
                labels = labels

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

            losses = tf.reduce_mean(losses, name="reduced_mean")

        with tf.name_scope("out_put_layer"):
            # [BATCH_SIZE]
            probabilities = tf.nn.softmax(logits, dim=-1)
            classes =  tf.equal(tf.round(probabilities), tf.round(labels))

            tf.logging.info('classes: =====> {}'.format(classes))
            tf.logging.info('probabilities: =====> {}'.format(probabilities))

            predictions = {
                self.feature_type.OUT_CLASSES: classes,
                # [BATCH_SIZE]
                self.feature_type.OUT_PROBABILITIES: probabilities
            }

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            global_step = tf.contrib.framework.get_global_step()
            learning_rate = self.bilstm_config.LEARNING_RATE
            #learning_rate = tf.train.exponential_decay(self.bilstm_config.LEARNING_RATE, global_step,
            #                                          100, 0.99, staircase=True)
            tf.summary.scalar(tensor=learning_rate, name="decaying_lr")
            train_op = tf.contrib.layers.optimize_loss(
                loss=losses,
                global_step=global_step,
                optimizer=tf.train.AdamOptimizer,
                learning_rate=learning_rate)

            loss = losses

            eval_metric_ops = {
                'MeanAccuracy' : tf.reduce_mean(tf.cast(predictions[self.feature_type.OUT_CLASSES],
                                                        tf.float32)),

                #     It will take the minimum of correct_prediction
                # for each element of the batch.This minimum is 1 if all elements are 1
                # (ie all predictions are correct) and 0 if at least one element is False ( and equal to 0)
                'AllLabelAccuracy' :  tf.reduce_mean(
                    tf.reduce_min(
                        tf.cast(predictions[self.feature_type.OUT_CLASSES]),
                        tf.float32), 1),
                'Accuracy': tf.metrics.accuracy(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions[self.feature_type.OUT_CLASSES],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions[self.feature_type.OUT_CLASSES],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions[self.feature_type.OUT_CLASSES],
                    name='Recall')
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            # training_hooks=self.hooks
        )


#Decay Leanring Rate Visulaization
# import math
# import matplotlib.pylot as plt
# steps_epoch = 15663//16
# num_epocs = 5
# learning_rate = 0.1
# decay_rate = 0.99
# decay_steps = 10
# global_step= range(0,steps_epoch*num_epocs, decay_steps)
# d_lr = [learning_rate *  math.pow(decay_rate, (step / decay_steps)) for step in global_step]
# plt.plot(d_lr)
