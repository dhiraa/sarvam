import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

import capslayer
from nlp.text_classification.dataset.feature_types import TextAndCharIdsFeature
from nlp.text_classification.tc_utils.tc_config import ModelConfigBase, EXPERIMENT_MODEL_ROOT_DIR

epsilon = 1e-9

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

class CapsulesTextClassifierV0Config(ModelConfigBase):
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
                 out_keep_propability,
                 batch_size):

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

        self.BATCH_SIZE = batch_size

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
        batch_size = input("batch_size: ") or -1

        while(batch_size == -1):
            batch_size = input("batch_size: ") or -1


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

        cfg =  CapsulesTextClassifierV0Config(model_dir,
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

        CapsulesTextClassifierV0Config.dump(model_dir, cfg)

        return cfg

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
run_config.allow_soft_placement = True
run_config.log_device_placement = False
run_config = tf.contrib.learn.RunConfig(session_config=run_config,
                                        save_checkpoints_steps=500,
                                        keep_checkpoint_max=3,
                                        save_summary_steps=50)

class CapsulesTextClassifierV0(tf.estimator.Estimator):
    feature_type = TextAndCharIdsFeature
    def __init__(self, config):
        super(CapsulesTextClassifierV0, self).__init__(
            model_fn=self._model_fn,
            model_dir=config.MODEL_DIR,
            config=run_config)

        self.capsule_config = config


    def _model_fn(self, features, labels, mode, params):
        """

        :param features: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param labels: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param mode: ModeKeys
        :param params:
        :return:
        """

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        token_ids = features[self.feature_type.FEATURE_1]

        # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
        char_ids = features[self.feature_type.FEATURE_2]

        labels = tf.cast(labels, tf.float32)

        print("\n\n\n\n\n")
        tf.logging.info('token_ids: =======> {}'.format(token_ids))
        tf.logging.info('char_ids: =======> {}'.format(char_ids))
        tf.logging.info('labels: =======> {}'.format(labels))

        s = tf.shape(char_ids)

        #remove pad words
        char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2])) #20 -> char dim

        with tf.variable_scope("word_embed_layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=self.capsule_config.WORD_VOCAB_SIZE,
                                                               embed_dim=self.capsule_config.WORD_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=self.capsule_config.KEEP_PROP,
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
                                                               vocab_size=self.capsule_config.CHAR_VOCAB_SIZE,
                                                               embed_dim=self.capsule_config.CHAR_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            #[BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.layers.dropout(char_embeddings,
                                                rate=self.capsule_config.KEEP_PROP,
                                                seed=42,
                                                training=mode == tf.estimator.ModeKeys.TRAIN)  # TODO add test case


            tf.logging.info('char_embeddings =====> {}'.format(char_embeddings))

        with tf.variable_scope("embeddings_reshape"):
            # put the time dimension on axis=1
            shape = tf.shape(char_embeddings)

            BATCH_SIZE = shape[0]
            MAX_DOC_LENGTH = shape[1]
            CHAR_MAX_LENGTH = shape[2]


            # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]  ===>
            #      [BATCH_SIZE * MAX_SEQ_LENGTH, MAX_WORD_LENGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.reshape(char_embeddings,
                                         shape=[BATCH_SIZE * MAX_DOC_LENGTH, CHAR_MAX_LENGTH,
                                                self.capsule_config.CHAR_EMBEDDING_SIZE, 1],
                                         name="reduce_dimension_1")

            word_embeddings = tf.expand_dims(word_embeddings, axis=-1)

        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(word_embeddings,
                                             num_outputs=256,
                                             kernel_size=9,
                                             stride=1,
                                             padding='VALID')

            # return primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = capslayer.layers.primaryCaps(conv1,
                                                                   filters=32,
                                                                   kernel_size=9,
                                                                   strides=2,
                                                                   out_caps_shape=[8, 1])

            # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
        with tf.variable_scope('DigitCaps_layer'):
            primaryCaps = tf.reshape(primaryCaps, shape=[self.capsule_config.BATCH_SIZE, -1, 8, 1])
            self.digitCaps, self.activation = capslayer.layers.fully_connected(primaryCaps,
                                                                               activation,
                                                                               num_outputs=10,
                                                                               out_caps_shape=[16, 1],
                                                                               routing_method='DynamicRouting')

        def loss(self):
            # 1. Margin loss

            # max_l = max(0, m_plus-||v_c||)^2
            max_l = tf.square(tf.maximum(0., cfg.m_plus - self.activation))
            # max_r = max(0, ||v_c||-m_minus)^2
            max_r = tf.square(tf.maximum(0., self.activation - cfg.m_minus))

            # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
            max_l = tf.reshape(max_l, shape=(self.capsule_config.BATCH_SIZE, -1))
            max_r = tf.reshape(max_r, shape=(self.capsule_config.BATCH_SIZE, -1))

            # calc T_c: [batch_size, num_label]
            # T_c = Y, is my understanding correct? Try it.
            T_c = self.Y
            # [batch_size, num_label], element-wise multiply
            L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

            # 2. The reconstruction loss
            orgin = tf.reshape(self.x, shape=(self.capsule_config.BATCH_SIZE, -1))
            squared = tf.square(self.decoded - orgin)
            self.reconstruction_err = tf.reduce_mean(squared)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            self.loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err


        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            global_step = tf.contrib.framework.get_global_step()
            learning_rate = self.capsule_config.LEARNING_RATE
            #learning_rate = tf.train.exponential_decay(cfg.learning_rate, global_step,
             #                                          100, 0.99, staircase=True)
            tf.summary.scalar(tensor=learning_rate, name="decaying_lr")
            train_op = tf.contrib.layers.optimize_loss(
                loss=self.total_loss,
                global_step=global_step,
                optimizer=tf.train.AdamOptimizer,
                learning_rate=learning_rate)

            loss = self.total_loss

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions["classes"],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions["classes"],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=tf.cast(tf.argmax(labels, axis=-1), tf.int32),
                    predictions=predictions["classes"],
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