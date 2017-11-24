import tensorflow as tf
import argparse
import math

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys

tf.logging.set_verbosity("INFO")


'''
Notes:
Two methods:
- CBOW: Continuous Bag of Word i.e given context words find the target word
- Skip Gram i.e given a word find the target context words

"The cat is sitting on the mat"

CBOW : is   | ( The      cat  )   ( sitting      on  )
       -----------------------------------------------
       w(t)    w(t-2)   w(t-1)       w(t+1)    w(t+2)
       
       Find centre word "is" given "The", "cat", "sitting" and "on"

Skip Gram:   ( The      cat  )   ( sitting      on  ) | is
             ---------------------------------------------
              w(t-2)   w(t-1)       w(t+1)    w(t+2)    w(t)
              
      Find context words of "is", here "The", "cat", "sitting" and "on"
      
      Features:
      (is, The)
      (is, cat)
      (is, sitting)
      (is, given)
      
      
Model: Word Embedding Matrix [Vocab Size, Embedding Size]       

'''

class Word2VecConfig():
    def __init__(self,
                 vocab_size,
                 words_vocab_file,
                 embedding_size,
                 word_emd_size,
                 num_word_sample,
                 learning_rate,
                 model_dir):
        tf.app.flags.FLAGS = tf.app.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()
        flags = tf.app.flags
        self.FLAGS = flags.FLAGS

        flags.DEFINE_string("UNKNOWN_WORD", "<UNK>", "")

        flags.DEFINE_integer("VOCAB_SIZE", vocab_size, "")
        flags.DEFINE_string("WORDS_VOCAB_FILE", words_vocab_file, "")

        flags.DEFINE_integer("EMBED_SIZE", embedding_size, "")
        flags.DEFINE_integer("NUM_WORD_SAMPLE", num_word_sample, "")

        flags.DEFINE_float("LEARNING_RATE", learning_rate, "")
        # flags.DEFINE_float("KEEP_PROP", out_keep_propability, "")
        flags.DEFINE_integer("WORD_EMBEDDING_SIZE", word_emd_size, "")

        flags.DEFINE_string("MODEL_DIR", model_dir, "")



class Word2Vec(tf.estimator.Estimator):
    '''
    Skip Gram implementation
    '''
    def __init__(self,
                 config:Word2VecConfig):
        super(Word2Vec, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.FLAGS.MODEL_DIR,
            config=None)

        self.w2v_config = config

    def _model_fn(self, features, labels, mode, params):

        center_words = features
        target_words = labels

        # Define model's architecture
        with tf.variable_scope("center-words-2-ids"):
            table = lookup.index_table_from_file(vocabulary_file=self.w2v_config.FLAGS.WORDS_VOCAB_FILE,
                                                 num_oov_buckets=0,
                                                 default_value=0,
                                                 name="table")
            tf.logging.info('table info: {}'.format(table))

            words = tf.string_split(center_words)
            densewords = tf.sparse_tensor_to_dense(words, default_value=self.w2v_config.FLAGS.UNKNOWN_WORD)
            center_word_ids = table.lookup(densewords)

            tf.logging.info("center_word_ids -----> {}".format(center_word_ids))
            #[batch_size,?] -> [batc_size, 1] -> [batch_size,]
            center_word_ids = tf.squeeze(tf.reshape(center_word_ids, shape=(-1, 1)))


        with tf.variable_scope("target-words-2-ids"):
            table = lookup.index_table_from_file(vocabulary_file=self.w2v_config.FLAGS.WORDS_VOCAB_FILE,
                                                 num_oov_buckets=0,
                                                 default_value=0,
                                                 name="table")
            tf.logging.info('table info: {}'.format(table))

            words = tf.string_split(center_words)
            densewords = tf.sparse_tensor_to_dense(words, default_value=self.w2v_config.FLAGS.UNKNOWN_WORD)
            target_word_ids = table.lookup(densewords)

            target_word_ids = tf.reshape(target_word_ids, shape=(-1, 1))

            tf.logging.info("target_word_ids -----> {}".format(target_word_ids))

        with tf.name_scope("embed"):
            embed_matrix = tf.Variable(tf.random_uniform([self.w2v_config.FLAGS.VOCAB_SIZE,
                                                          self.w2v_config.FLAGS.EMBED_SIZE], -1.0, 1.0),
                                       name="embed_matrix")

            tf.logging.info("embed_matrix -----> {}".format(embed_matrix))


        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(embed_matrix, center_word_ids, name="embed")

            tf.logging.info("embed -----> {}".format(embed))


            # construct variables for NCE loss
            nce_weight = tf.Variable(tf.truncated_normal([self.w2v_config.FLAGS.VOCAB_SIZE, self.w2v_config.FLAGS.EMBED_SIZE],
                                                         stddev= 1/math.sqrt( self.w2v_config.FLAGS.EMBED_SIZE)),
                                     name="nce_weight")

            tf.logging.info("nce_weight -----> {}".format(nce_weight))

            nce_bias = tf.Variable(tf.zeros([ self.w2v_config.FLAGS.VOCAB_SIZE]), name="nce_bias")
            tf.logging.info("nce_bias -----> {}".format(nce_bias))


        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            # define loss function to be NCE loss function
            # define loss function to be NCE loss function
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                 biases=nce_bias,
                                                 labels=target_word_ids,
                                                 inputs=embed,
                                                 num_sampled=self.w2v_config.FLAGS.NUM_WORD_SAMPLE,
                                                 num_classes=self.w2v_config.FLAGS.VOCAB_SIZE), name='loss')

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                optimizer=tf.train.GradientDescentOptimizer,
                learning_rate=self.w2v_config.FLAGS.LEARNING_RATE)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=None
        )



