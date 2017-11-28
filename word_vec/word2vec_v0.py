import tensorflow as tf
import argparse
import math
import numpy as np

from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.tensorboard.plugins import projector

tf.logging.set_verbosity("INFO")

from word_vec.utils.tf_hooks.post_run import PostRunTaskHook
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
                 num_word_sample,
                 learning_rate,
                 model_dir,
                 cooccurrence_cap,
                 scaling_factor):
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

        flags.DEFINE_string("MODEL_DIR", model_dir, "")

        flags.DEFINE_integer("COOCCURRENCE_CAP", cooccurrence_cap, "")
        flags.DEFINE_integer("SCALING_FACTOR", scaling_factor, "")


class Word2Vec(tf.estimator.Estimator):
    '''
    Skip Gram implementation
    '''
    def __init__(self,
                 config:Word2VecConfig):
        super(Word2Vec, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.FLAGS.MODEL_DIR,
            config=tf.contrib.learn.RunConfig(log_step_count_steps=100,
                                              save_summary_steps=100,
                                              gpu_memory_fraction=0.5,
                                              save_checkpoints_steps=1000,
                                              tf_random_seed=42,
                                              log_device_placement=True))

        self.w2v_config = config

        self.embed_mat_hook = None #Hook to store the embedding matrix as numpy utils

    def _model_fn(self, features, labels, mode, params):

        with tf.variable_scope("inputs"):
            focal_input = features["focal_words"]
            context_input = features["context_words"]
            cooccurrence_count = features["cooccurrence_count"]

            count_max = tf.constant([self.w2v_config.FLAGS.COOCCURRENCE_CAP],
                                    dtype=tf.float32,
                                    name='max_cooccurrence_count')

            scaling_factor = tf.constant([self.w2v_config.FLAGS.SCALING_FACTOR],
                                         dtype=tf.float32,
                                         name="scaling_factor")


        with tf.variable_scope("model_parameters"):
            #[vocab_size x embedding_size] w
            focal_embeddings = tf.Variable(tf.random_uniform([self.w2v_config.FLAGS.VOCAB_SIZE,
                                                              self.w2v_config.FLAGS.EMBED_SIZE],
                                                             1.0, -1.0),
                                            name="focal_embeddings")

            # [vocab_size x embedding_size] w˜
            context_embeddings = tf.Variable(tf.random_uniform([self.w2v_config.FLAGS.VOCAB_SIZE,
                                                                self.w2v_config.FLAGS.EMBED_SIZE],
                                                               1.0, -1.0),
                                                name="context_embeddings")

            # [vocab_size]
            # b_i {i : 0 to Vocab Size}
            focal_biases = tf.Variable(tf.random_uniform([self.w2v_config.FLAGS.VOCAB_SIZE],
                                                         1.0, -1.0),
                                       name='focal_biases')

            # b_j˜ {j : 0 to Vocab Size}
            context_biases = tf.Variable(tf.random_uniform([self.w2v_config.FLAGS.VOCAB_SIZE],
                                                           1.0, -1.0),
                                         name="context_biases")

        with tf.variable_scope("embedding_lookup"):
            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], context_input)

        with tf.variable_scope("embedding"):
            # f(x) = min(1, (count/count_max)^scaling_factor)
            weighting_factor = tf.minimum(1.0, tf.pow(tf.div(cooccurrence_count, count_max),
                                                    scaling_factor))

            # W^t W˜
            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            # log(X_ij)
            log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

            # W^t W˜ + b_i + b_j˜ - log(X_ij)
            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)

            combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            # define loss function to be NCE loss function
            # J
            loss = tf.reduce_sum(single_losses, name="GloVe_loss")

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                optimizer=tf.train.AdagradOptimizer,
                learning_rate=self.w2v_config.FLAGS.LEARNING_RATE)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=None
        )

    def set_store_hook(self, tensor_name="embedding/combined_embeddings:0"):
        def save_embed_mat(sess):
            graph = sess.graph
            embed_mat = graph.get_tensor_by_name(tensor_name)

            embed_mat = sess.run(embed_mat)
            np.save("tmp/word2vec_v0.npy", embed_mat)

        self.embed_mat_hook = PostRunTaskHook()
        self.embed_mat_hook.user_func = save_embed_mat


    def get_store_hook(self):
        self.set_store_hook()
        return self.embed_mat_hook



