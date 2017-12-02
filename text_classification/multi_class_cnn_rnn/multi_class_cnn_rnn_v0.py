from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from utils.rnn import *
import argparse

tf.logging.set_verbosity(tf.logging.DEBUG)

class MultiClassCNNRNNConfig():
    def __init__(self,
                 vocab_size,
                 model_dir,
                 words_vocab_file,
                 learning_rate=0.01,
                 word_level_lstm_hidden_size=128,
                 word_emd_size=128,
                 out_keep_propability=0.9):
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()
        flags = tf.app.flags
        self.FLAGS = flags.FLAGS

        # Constant params
        flags.DEFINE_string("MODEL_DIR", model_dir, "")

        # Preprocessing Paramaters
        flags.DEFINE_string("WORDS_VOCAB_FILE", words_vocab_file, "")
        flags.DEFINE_string("UNKNOWN_WORD", "<UNK>", "")

        flags.DEFINE_integer("VOCAB_SIZE", vocab_size, "")

        # Model hyper paramaters
        flags.DEFINE_float("LEARNING_RATE", learning_rate, "")
        flags.DEFINE_float("KEEP_PROP", out_keep_propability, "")
        flags.DEFINE_integer("WORD_EMBEDDING_SIZE", word_emd_size, "")
        flags.DEFINE_integer("WORD_LEVEL_LSTM_HIDDEN_SIZE", word_level_lstm_hidden_size, "")

    def get_tf_flag(self):
        # usage config.FLAGS.MODEL_DIR
        return self.FLAGS


class MultiClassCNNRNNV0(tf.estimator.Estimator):
    def __init__(self,
                 config: MultiClassCNNRNNConfig):
        super(MultiClassCNNRNNV0, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.FLAGS.MODEL_DIR,
            config=None)

        self.VOCAB_FILE = config.FLAGS.WORDS_VOCAB_FILE
        self.VOCAB_SIZE = config.FLAGS.VOCAB_SIZE
        self.UNKNOWN_WORD = config.FLAGS.UNKNOWN_WORD
        self.EMBEDDING_SIZE = 40

        self.WINDOW_SIZE = self.EMBEDDING_SIZE
        self.STRIDE = int(self.WINDOW_SIZE / 2)

        self.NUM_CLASSES = 3

        self.LEARNING_RATE = config.FLAGS.LEARNING_RATE

        self.num_lstm_layers = 4
        self.output_keep_prob = 0.9

        # self.VOCAB_FILE = config.FLAGS.WORDS_VOCAB_FILE
        # self.VOCAB_SIZE = config.FLAGS.VOCAB_SIZE
        # self.UNKNOWN_WORD = config.FLAGS.UNKNOWN_WORD

        self.PADWORD = 'PADXYZ'
        # self.MAX_DOCUMENT_LENGTH = max_doc_length
        # self.EMBEDDING_SIZE = 32

        self.RNN_HIDDEL_SIZE = self.EMBEDDING_SIZE
        # self.WINDOW_SIZE = self.EMBEDDING_SIZE
        # self.STRIDE = int(self.WINDOW_SIZE / 2)
        #
        # self.NUM_CLASSES = 3
        #
        # self.num_lstm_layers = 2
        # self.output_keep_prob = 0.5

    #
    # def _model_fn(self, features, labels, mode, params):
    #     """Model function used in the estimator.
    #
    #     Args:
    #         features : Tensor(shape=[?], dtype=string) Input features to the model.
    #         labels : Tensor(shape=[?, n], dtype=Float) Input labels.
    #         mode (ModeKeys): Specifies if training, evaluation or prediction.
    #         params (HParams): hyperparameters.
    #
    #     Returns:
    #         (EstimatorSpec): Model to be run by Estimator.
    #     """
    #     is_training = mode == ModeKeys.TRAIN
    #
    #     text_features = features["text"]
    #
    #     tf.logging.debug('text_features -----> {}'.format(text_features))
    #     tf.logging.debug('labels -----> {}'.format(labels))
    #
    #     # Define model's architecture
    #     with tf.variable_scope("sentence-2-words"):
    #         table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE,
    #                                              num_oov_buckets=0,
    #                                              default_value=0,
    #                                              name="table")
    #         tf.logging.info('table info: {}'.format(table))
    #
    #         words = tf.string_split(text_features)
    #         densewords = tf.sparse_tensor_to_dense(words, default_value=self.UNKNOWN_WORD)
    #         token_ids = table.lookup(densewords)
    #
    #
    #     with tf.device('/cpu:0'), tf.name_scope("embed-layer"):
    #         # layer to take the words and convert them into vectors (embeddings)
    #         # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    #         # maps word indexes of the sequence into
    #         # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
    #         word_vectors = tf.contrib.layers.embed_sequence(token_ids,
    #                                                         vocab_size=self.VOCAB_SIZE,
    #                                                         embed_dim=self.EMBEDDING_SIZE,
    #                                                         initializer=tf.contrib.layers.xavier_initializer(seed=42))
    #
    #         # [?, self.MAX_DOCUMENT_LENGTH, self.EMBEDDING_SIZE]
    #         tf.logging.info('words_embed={}'.format(word_vectors))
    #
    #     with tf.device("/gpu:0"), tf.name_scope("fast_text"):
    #
    #         #[?, self.EMBEDDING_SIZE]
    #         averaged_word_vectors = tf.reduce_sum(word_vectors, axis=1)
    #
    #         tf.logging.info('words_embed={}'.format(averaged_word_vectors))
    #
    #     with  tf.name_scope("hidden-mlp-layer"):
    #         #Wide layer connecting the input ids to the hidden layer
    #         hidden_layer =  tf.layers.dense(inputs=averaged_word_vectors,
    #                                         units=1024,
    #                                         activation=tf.nn.relu,
    #                                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
    #
    #         hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
    #                                          training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    #         hidden_layer =  tf.layers.dense(inputs=hidden_layer,
    #                                         units=self.EMBEDDING_SIZE/2,
    #                                         activation=tf.nn.relu,
    #                                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
    #
    #         hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
    #                                      training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    #         tf.logging.info('wide_layer: ------> {}'.format(hidden_layer))
    #
    #     with  tf.name_scope("logits-layer"):
    #         # [?, self.NUM_CLASSES]
    #
    #         logits = tf.layers.dense(inputs=hidden_layer, units=self.NUM_CLASSES,
    #                                  kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
    #
    #         tf.logging.info('logits: ------> {}'.format(logits))
    #
    #     with  tf.name_scope("output-layer"):
    #         # [?,1]
    #         predicted_class = tf.argmax(logits, axis=1, name="class_output")
    #         tf.logging.info('predicted_class: ------> {}'.format(predicted_class))
    #
    #         predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
    #         tf.logging.info('predicted_probabilities: ------> {}'.format(predicted_probabilities))
    #
    #     predictions = {
    #         "classes": predicted_class,
    #         "probabilities": predicted_probabilities
    #     }
    #
    #     #logging
    #     # self.log_tensors("output_probabilities", "output-layer/softmax_output")
    #     tf.summary.histogram(averaged_word_vectors.name, averaged_word_vectors)
    #     tf.summary.histogram(predicted_probabilities.name, predicted_probabilities)
    #
    #     # Loss, training and eval operations are not needed during inference.
    #     loss = None
    #     train_op = None
    #     eval_metric_ops = {}
    #
    #     if mode != ModeKeys.INFER:
    #         loss = tf.losses.softmax_cross_entropy(
    #             onehot_labels=labels,
    #             logits=logits,
    #             weights=0.80,
    #             scope='actual_loss')
    #
    #         loss = tf.reduce_mean(loss, name="reduced_mean")
    #
    #         train_op = tf.contrib.layers.optimize_loss(
    #             loss=loss,
    #             global_step=tf.contrib.framework.get_global_step(),
    #             optimizer=tf.train.AdamOptimizer,
    #             learning_rate=self.LEARNING_RATE )
    #
    #         label_argmax = tf.argmax(labels, 1, name='label_argmax')
    #
    #         eval_metric_ops = {
    #             'Accuracy': tf.metrics.accuracy(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='accuracy'),
    #             'Precision' : tf.metrics.precision(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='Precision'),
    #             'Recall' : tf.metrics.recall(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='Recall')
    #         }
    #
    #
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         loss=loss,
    #         train_op=train_op,
    #         eval_metric_ops=eval_metric_ops
    #     )



    def _model_fn(self, features, labels, mode, params):
        """Model function used in the estimator.

        Args:
            features : Tensor(shape=[?], dtype=string) Input features to the model.
            labels : Tensor(shape=[?, n], dtype=Float) Input labels.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.

        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == ModeKeys.TRAIN

        text_features = features["text"]
        # numeric_features = features["numeric"]

        tf.logging.debug('text_features -----> {}'.format(text_features))
        # tf.logging.debug('numeric_features -----> {}'.format(numeric_features))
        tf.logging.debug('labels -----> {}'.format(labels))

        # Define model's architecture
        with tf.variable_scope("sentence-2-words"):
            table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE,
                                                 num_oov_buckets=1,
                                                 default_value=-1,
                                                 name="table")
            tf.logging.info('table info: {}'.format(table))

            words = tf.string_split(text_features)
            densewords = tf.sparse_tensor_to_dense(words, default_value=self.PADWORD)
            token_ids = table.lookup(densewords)

            # padding = tf.constant([[0, 0], [0, self.MAX_DOCUMENT_LENGTH]])
            # padded = tf.pad(token_ids, padding)
            # sliced = tf.slice(padded, [0, 0], [-1, self.MAX_DOCUMENT_LENGTH])

            # sliced = tf.cast(sliced, tf.float32)
            # sliced = tf.contrib.layers.batch_norm(sliced, center=True, scale=True,
            #                                       is_training=is_training,
            #                                       scope='bn')

            # tf.logging.info('sliced -----> {}'.format(sliced))

        with tf.device('/cpu:0'), tf.name_scope("embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
            word_vectors = tf.contrib.layers.embed_sequence(token_ids,
                                                            vocab_size=self.VOCAB_SIZE,
                                                            embed_dim=self.EMBEDDING_SIZE)

            # [?, self.MAX_DOCUMENT_LENGTH, self.EMBEDDING_SIZE]
            tf.logging.debug('words_embed={}'.format(word_vectors))

        with  tf.name_scope("lstm-layer"):
            # LSTM cell
            # cell = tf.contrib.rnn.LSTMCell(self.EMBEDDING_SIZE, state_is_tuple=True)

            # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
            cell = tf.nn.rnn_cell.GRUCell(self.RNN_HIDDEL_SIZE)
            tf.logging.info('cell: ------> {}'.format(cell))

            if is_training:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
            else:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0)

            # Stack up multiple LSTM layers, for deep learning
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_lstm_layers)
            tf.logging.info('cell: ------> {}'.format(cell))
            #
            outputs, encoding = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32,
                                                  sequence_length=get_sequence_length(word_vectors))
            # LSTMStateTuple https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
            # encoding = encoding[0][0]

            # tf.nn.bidirectional_dynamic_rnn()

            # GRUCell
            encoding = encoding[0]

            # Split into list of embedding per word, while removing doc length dim.
            # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
            # word_vectors = tf.unstack(word_vectors, axis=1)
            #
            # # Create an unrolled Recurrent Neural Networks to length of
            # # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
            # _, encoding = tf.nn.static_rnn(cell, word_vectors, dtype=tf.float32)
            # [?, EMBEDDING_SIZE]
            tf.logging.info('encoding: ------> {}'.format(encoding))

        with  tf.name_scope("hidden-mlp-layer"):
            # Wide layer connecting the input ids to the hidden layer
            hidden_layer = tf.layers.dense(inputs=encoding, units=32 * 32, activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            hidden_layer = tf.layers.dropout(hidden_layer, rate=0.9, seed=42,
                                             training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('wide_layer: ------> {}'.format(hidden_layer))
            #
            # hidden_layer =  tf.layers.dense(inputs=hidden_layer, units=768, activation=tf.nn.relu,
            #                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
            #
            # tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))
            #
            # hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
            #                                  training=mode == tf.estimator.ModeKeys.TRAIN)
            #
            #
            # hidden_layer =  tf.layers.dense(inputs=hidden_layer, units=512, activation=tf.nn.relu,
            #                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
            #
            # tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))
            #
            # hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
            #                             training=mode == tf.estimator.ModeKeys.TRAIN)

        with  tf.name_scope('conv_layer'):

            encoding_2_image = tf.reshape(hidden_layer, [-1, 32, 32, 1])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=encoding_2_image,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            tf.logging.info('conv1: ------> {}'.format(conv1))  # [?, 32, 32, 32]

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # [?,16, 16, 32]
            tf.logging.info('pool1: ------> {}'.format(pool1))

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
            tf.logging.info('conv2: ------> {}'.format(conv2))  # [?, 16, 16, 64]

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            tf.logging.info('pool2: ------> {}'.format(pool2))  # [?, 8, 8, 64]

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
            tf.logging.info('pool2_flat: ------> {}'.format(pool2_flat))  # [?, 8, 8, 64]

            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            hidden_layer = tf.layers.dropout(
                inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)

        with  tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]

            logits = tf.layers.dense(inputs=hidden_layer, units=self.NUM_CLASSES,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            # logits = tf.layers.dense(inputs=tf.concat([hidden_layer, numeric_features], axis=1), units=self.NUM_CLASSES,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('logits: ------> {}'.format(logits))

        with  tf.name_scope("output-layer"):
            # [?,1]
            predicted_class = tf.argmax(logits, axis=1, name="class_output")
            tf.logging.info('predicted_class: ------> {}'.format(predicted_class))

            predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
            tf.logging.info('predicted_probabilities: ------> {}'.format(predicted_probabilities))

        predictions = {
            "classes": predicted_class,
            "probabilities": predicted_probabilities
        }

        # logging
        # self.log_tensors("output_probabilities", "output-layer/softmax_output")
        tf.summary.histogram(encoding.name, encoding)
        tf.summary.histogram(predicted_probabilities.name, predicted_probabilities)

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            tf.logging.info('labels: ------> {}'.format(labels))
            tf.logging.info('predictions["classes"]: ------> {}'.format(predictions["classes"]))

            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels,
                logits=logits,
                weights=0.80,
                scope='actual_loss')

            loss = tf.reduce_mean(loss, name="reduced_mean")

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                optimizer=tf.train.AdamOptimizer,
                learning_rate=self.LEARNING_RATE)

            label_argmax = tf.argmax(labels, 1, name='label_argmax')

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Recall')
            }
            tf.summary.scalar(loss.name, loss)
            # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            #     test_set.utils,
            #     test_set.target,
            #     every_n_steps=50,
            #     metrics=validation_metrics,
            #     early_stopping_metric="loss",
            #     early_stopping_metric_minimize=True,
            #     early_stopping_rounds=200)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )
