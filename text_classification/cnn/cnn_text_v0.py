import os
import sys

# add utils path
sys.path.append("../")

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from utils.rnn import *
import argparse

tf.logging.set_verbosity(tf.logging.DEBUG)

class CNNTextV0Config():
    def __init__(self,
                 vocab_size,
                 model_dir,
                 words_vocab_file,
                 num_classes=3,
                 learning_rate=0.001,
                 word_emd_size = 48,
                 out_keep_propability=0.5):
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()
        flags = tf.app.flags
        self.FLAGS = flags.FLAGS

        #Constant params
        flags.DEFINE_string("MODEL_DIR", model_dir, "")

        #Preprocessing Paramaters
        flags.DEFINE_string("WORDS_VOCAB_FILE", words_vocab_file, "")
        flags.DEFINE_string("UNKNOWN_WORD", "<UNK>", "")

        flags.DEFINE_integer("VOCAB_SIZE", vocab_size, "")

        #Model hyper paramaters
        flags.DEFINE_float("LEARNING_RATE", learning_rate, "")
        flags.DEFINE_float("NUM_CLASSES", num_classes, "Number of output classes/category")
        flags.DEFINE_float("KEEP_PROP", out_keep_propability, "")
        flags.DEFINE_integer("WORD_EMBEDDING_SIZE", word_emd_size, "")

    def get_tf_flag(self):
        # usage config.FLAGS.MODEL_DIR
        return self.FLAGS

class CNNTextV0(tf.estimator.Estimator):
    def __init__(self,
                 config: CNNTextV0Config):
        super(CNNTextV0, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.FLAGS.MODEL_DIR,
            config=None)

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
