import tensorflow as tf
import argparse
import os

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from nlp.text_classification.tc_utils.feature_types import TextFeature
from  nlp.text_classification.tc_utils.tc_config import *

tf.logging.set_verbosity(tf.logging.DEBUG)

class FastTextV0Config(ModelConfigBase):
    def __init__(self,
                 vocab_size,
                 model_dir,
                 words_vocab_file,
                 num_classes,
                 learning_rate,
                 word_emd_size,
                 out_keep_propability):
        # tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        # tf.app.flags._global_parser = argparse.ArgumentParser()
        # flags = tf.app.flags
        # self.FLAGS = flags.FLAGS

        #Constant params
        self.MODEL_DIR =  model_dir

        #Preprocessing Paramaters
        self.WORDS_VOCAB_FILE =  words_vocab_file
        self.UNKNOWN_WORD = "<UNK>"
        self.VOCAB_SIZE = vocab_size

        #Model hyper paramaters
        self.LEARNING_RATE = learning_rate
        self.NUM_CLASSES = num_classes
        self.KEEP_PROP = out_keep_propability
        self.WORD_EMBEDDING_SIZE = word_emd_size


    @staticmethod
    def user_config(dataframe, data_iterator_name):
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

        model_dir = EXPERIMENT_MODEL_ROOT_DIR + dataframe.dataset_name +"/" + data_iterator_name+ "/FastTextV0/" + model_dir

        cfg = FastTextV0Config(vocab_size,
                 model_dir,
                 words_vocab_file,
                 num_classes,
                 learning_rate,
                 word_emd_size,
                 out_keep_propability)

        FastTextV0Config.dump(model_dir, cfg)

        return cfg

class FastTextV0(tf.estimator.Estimator):

    feature_type = TextFeature

    def __init__(self,
                 config):
        super(FastTextV0, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.MODEL_DIR,
            config=None)

        self.VOCAB_FILE = config.WORDS_VOCAB_FILE
        self.VOCAB_SIZE = config.VOCAB_SIZE
        self.UNKNOWN_WORD = config.UNKNOWN_WORD
        self.EMBEDDING_SIZE = config.WORD_EMBEDDING_SIZE

        self.NUM_CLASSES = config.NUM_CLASSES

        self.LEARNING_RATE = config.LEARNING_RATE

    def _model_fn(self, features, labels, mode, params):
        """Model function used in the estimator.

        Args:
            features : Tensor(shape=[?], dtype=string) Input features to the model.
            labels : Tensor(shape=[?, n], dtype=Float) One hot encoded Input labels.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.

        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == ModeKeys.TRAIN

        # text_features = features["text"]
        token_ids = features[self.feature_type.FEATURE_1]
        tf.logging.debug('text_features -----> {}'.format(token_ids))
        tf.logging.debug('labels -----> {}'.format(labels))

        # # Define model's architecture
        # with tf.variable_scope("sentence-2-words"):
        #     table = lookup.index_table_from_file(vocabulary_file=self.VOCAB_FILE,
        #                                          num_oov_buckets=0,
        #                                          default_value=0,
        #                                          name="table")
        #     tf.logging.info('table info: {}'.format(table))
        #
        #     words = tf.string_split(text_features)
        #     densewords = tf.sparse_tensor_to_dense(words, default_value=self.UNKNOWN_WORD)
        #     token_ids = table.lookup(densewords)


        with tf.device('/cpu:0'), tf.name_scope("embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
            word_vectors = tf.contrib.layers.embed_sequence(token_ids,
                                                            vocab_size=self.VOCAB_SIZE,
                                                            embed_dim=self.EMBEDDING_SIZE,
                                                            initializer=tf.contrib.layers.xavier_initializer(seed=42))

            # [?, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE]
            tf.logging.info('words_embed={}'.format(word_vectors))

        with tf.device("/gpu:0"), tf.name_scope("fast_text"):

            #[?, EMBEDDING_SIZE]
            averaged_word_vectors = tf.reduce_sum(word_vectors, axis=1)

            tf.logging.info('words_embed={}'.format(averaged_word_vectors))

        with  tf.name_scope("hidden-mlp-layer"):
            #Wide layer connecting the input ids to the hidden layer
            hidden_layer =  tf.layers.dense(inputs=averaged_word_vectors,
                                            units=1024,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
                                             training=mode == tf.estimator.ModeKeys.TRAIN)

            hidden_layer =  tf.layers.dense(inputs=hidden_layer,
                                            units=self.EMBEDDING_SIZE/2,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            hidden_layer = tf.layers.dropout(hidden_layer, rate=0.5, seed=42,
                                         training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('wide_layer: ------> {}'.format(hidden_layer))

        with  tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]

            logits = tf.layers.dense(inputs=hidden_layer, units=self.NUM_CLASSES,
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

        #logging
        # self.log_tensors("output_probabilities", "output-layer/softmax_output")
        tf.summary.histogram(averaged_word_vectors.name, averaged_word_vectors)
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
                learning_rate=self.LEARNING_RATE )

            label_argmax = tf.argmax(labels, 1, name='label_argmax')

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='accuracy'),
                'Precision' : tf.metrics.precision(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Precision'),
                'Recall' : tf.metrics.recall(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Recall')
            }

            tf.summary.scalar(loss.name, loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )


