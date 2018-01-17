import os
import sys

# add audio_utils path
from nlp.text_classification.tc_utils.feature_types import TextFeature
from nlp.text_classification.tc_utils.tc_config import ModelConfigBase, EXPERIMENT_MODEL_ROOT_DIR

sys.path.append("../")

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from nlp.text_classification.tc_utils.rnn import *
import argparse
from sarvam.config.global_constants import *

tf.logging.set_verbosity(tf.logging.DEBUG)

class CNNTextV0Config(ModelConfigBase):
    def __init__(self,
                 max_doc_length,
                 vocab_size,
                 model_dir,
                 words_vocab_file,
                 num_classes,
                 learning_rate,
                 word_emd_size,
                 out_keep_propability):
        #Constant params
        self.MODEL_DIR =  model_dir

        #Preprocessing Paramaters
        self.WORDS_VOCAB_FILE =  words_vocab_file
        self.VOCAB_SIZE = vocab_size
        self.MAX_DOC_LEGTH = max_doc_length

        #Model hyper paramaters
        self.LEARNING_RATE = learning_rate
        self.NUM_CLASSES = num_classes
        self.KEEP_PROP = out_keep_propability
        self.WORD_EMBEDDING_SIZE = word_emd_size


    @staticmethod
    def user_config(dataframe, data_iterator_name):
        max_doc_length = dataframe.MAX_DOC_LEGTH
        vocab_size = dataframe.WORD_VOCAB_SIZE

        words_vocab_file = dataframe.words_vocab_file
        num_classes = dataframe.NUM_CLASSES
        learning_rate = input("learning_rate: (0.001): ") or 0.001
        word_emd_size = input("word_emd_size (32): ") or 32
        out_keep_propability = input("out_keep_propability (0.5): ") or 0.5

        model_dir = "lr_{}_wemd_{}_keep_{}".format(
            learning_rate,
            word_emd_size,
            out_keep_propability
        )

        model_dir = EXPERIMENT_MODEL_ROOT_DIR + dataframe.dataset_name +"/" + data_iterator_name+ "/CNNTextV0/" + model_dir

        cfg = CNNTextV0Config(max_doc_length,
                              vocab_size,
                              model_dir,
                              words_vocab_file,
                              num_classes,
                              learning_rate,
                              word_emd_size,
                              out_keep_propability)

        CNNTextV0Config.dump(model_dir, cfg)

        return cfg

class CNNTextV0(tf.estimator.Estimator):

    feature_type = TextFeature

    def __init__(self,
                 config: CNNTextV0Config):
        super(CNNTextV0, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.MODEL_DIR,
            config=None)
        self.cnn_config = config

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

        token_ids = features[self.feature_type.FEATURE_1]
        tf.logging.debug('token_ids -----> {}'.format(token_ids))
        tf.logging.debug('labels -----> {}'.format(labels))

        # Define model's architecture
        # with tf.variable_scope("sentence-2-words"):
        #     table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE,
        #                                          num_oov_buckets=1,
        #                                          default_value=-1,
        #                                          name="table")
        #     tf.logging.info('table info: {}'.format(table))
        #
        #     words = tf.string_split(text_features)
        #     densewords = tf.sparse_tensor_to_dense(words, default_value=self.PADWORD)
        #     token_ids = table.lookup(densewords)
        #
        #     padding = tf.constant([[0, 0], [0, self.MAX_DOCUMENT_LENGTH]])
        #     padded = tf.pad(token_ids, padding)
        #     sliced = tf.slice(padded, [0, 0], [-1, self.MAX_DOCUMENT_LENGTH])
        #
        #     sliced = tf.cast(sliced, tf.float32)
        #     sliced = tf.contrib.layers.batch_norm(sliced, center=True, scale=True,
        #                                           is_training=is_training,
        #                                           scope='bn')

        # tf.logging.info('sliced -----> {}'.format(sliced))

        with tf.device('/cpu:0'), tf.name_scope("embed_layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
            word_vectors = tf.contrib.layers.embed_sequence(token_ids,
                                                            vocab_size = self.cnn_config.VOCAB_SIZE,
                                                            embed_dim = self.cnn_config.WORD_EMBEDDING_SIZE)

            # [?, self.MAX_DOCUMENT_LENGTH, self.EMBEDDING_SIZE]
            tf.logging.debug('words_embed={}'.format(word_vectors))

        with  tf.name_scope('conv_layer'):

            encoding_2_image = tf.reshape(word_vectors, [-1,
                                                         self.cnn_config.MAX_DOC_LEGTH, #100
                                                         self.cnn_config.WORD_EMBEDDING_SIZE, #32
                                                         1])

            tf.logging.info('encoding_2_image: ------> {}'.format(encoding_2_image))

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=encoding_2_image,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            tf.logging.info('conv1: ------> {}'.format(conv1))  # [?, 100, 32, 32]

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # [?, 50, 16, 32]
            tf.logging.info('pool1: ------> {}'.format(pool1))

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
            tf.logging.info('conv2: ------> {}'.format(conv2))  # [?, 50, 16, 64]

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            tf.logging.info('pool2: ------> {}'.format(pool2))  # [?, 25, 8, 64]

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 25 * 8 * 64])
            tf.logging.info('pool2_flat: ------> {}'.format(pool2_flat))  # [?, 25, 8, 64]

            dense = tf.layers.dense(inputs=pool2_flat, units=25 * 8 * 64, activation=tf.nn.relu)

            hidden_layer = tf.layers.dropout(
                inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))

        with  tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]

            logits = tf.layers.dense(inputs=hidden_layer,
                                     units=self.cnn_config.NUM_CLASSES,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

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
        # tf.summary.histogram(encoding.name, encoding)
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
                learning_rate=self.cnn_config.LEARNING_RATE)

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
            #     test_set.audio_utils,
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
