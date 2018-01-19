import tensorflow as tf
import argparse
import os

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from nlp.text_classification.tc_utils.feature_types import TextIdsFeature
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

    feature_type = TextIdsFeature

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