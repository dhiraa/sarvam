import tensorflow as tf
import argparse

def get_capsules_config():

    tf.app.flags.FLAGS = tf.app.flags._FlagValues()
    tf.app.flags._global_parser = argparse.ArgumentParser()
    flags = tf.app.flags
    ############################
    #    hyper parameters      #
    ############################

    # For separate margin loss
    flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
    flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
    flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

    # for training
    flags.DEFINE_integer('num_classes', 3, 'number of output classes')
    flags.DEFINE_integer('batch_size', 16, 'batch size')
    flags.DEFINE_integer('epoch', 50, 'epoch')
    flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
    flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

    flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
    flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

    # LSTM params
    flags.DEFINE_integer('word_embedding_size', 24, 'word embedding size')
    flags.DEFINE_integer('char_embedding_size', 48, 'word embedding size')
    flags.DEFINE_integer('word_level_lstm_hidden_size', 48, 'lst hidden size for words')
    flags.DEFINE_integer('char_level_lstm_hidden_size', 24, 'lst hidden size for words')
    flags.DEFINE_integer('num_word_lstm_layers', 1, 'number of word lstm layers')
    flags.DEFINE_integer('output_keep_probability', 0.5, 'output keep probability')



    ############################
    #   environment setting    #
    ############################
    flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
    flags.DEFINE_boolean('is_training', True, 'train or predict phase')
    flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
    flags.DEFINE_string('model_dir', 'tmp/', 'model directory')
    flags.DEFINE_string('logdir', 'logdir', 'logs directory')
    flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
    flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
    flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
    flags.DEFINE_string('results', 'results', 'path for saving results')

    ############################
    #   distributed setting    #
    ############################
    flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
    flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
    flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

    cfg = tf.app.flags.FLAGS
    return cfg
# tf.logging.set_verbosity(tf.logging.INFO)
