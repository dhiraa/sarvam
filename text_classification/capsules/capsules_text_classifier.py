import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from capsules.capsules_config import get_capsules_config
from capsules.capsLayer import CapsLayer


epsilon = 1e-9

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
run_config.allow_soft_placement = True
run_config.log_device_placement = False
run_config=tf.contrib.learn.RunConfig(session_config=run_config,
                                      save_checkpoints_steps=100,
                                      keep_checkpoint_max=100)
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

class CapsulesTextClassifierV0(tf.estimator.Estimator):
    def __init__(self,
                 word_vocab_size,
                 char_vocab_size,
                 max_doc_length,
                 max_word_length,
                 num_classes):
        super(CapsulesTextClassifierV0, self).__init__(
            model_fn=self._model_fn,
            model_dir="tmp/CapsulesTextClassifierV0/",
            config=run_config)


        self.WORD_VOCAB_SIZE = word_vocab_size
        self.CHAR_VOCAB_SIZE = char_vocab_size
        self.MAX_DOC_LENGTH = max_doc_length
        self.MAX_CHAR_LENGTH = max_word_length
        self.NUM_CLASSES = num_classes

    def _model_fn(self, features, labels, mode, params):
        '''

        :param features: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param labels: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param mode: ModeKeys
        :param params:
        :return:
        '''

        cfg = get_capsules_config()

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        token_ids = features['word_ids']

        # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
        char_ids = features['char_ids']

        labels = tf.cast(labels, tf.float32)

        print("\n\n\n\n\n")
        tf.logging.info('token_ids: =======> {}'.format(token_ids))
        tf.logging.info('char_ids: =======> {}'.format(char_ids))
        tf.logging.info('labels: =======> {}'.format(labels))

        s = tf.shape(char_ids)

        #remove pad words
        char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2])) #20 -> char dim

        with tf.variable_scope("word-embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=self.WORD_VOCAB_SIZE,
                                                               embed_dim=cfg.word_embedding_size,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=cfg.output_keep_probability,
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
                                                               vocab_size=self.CHAR_VOCAB_SIZE,
                                                               embed_dim=cfg.char_embedding_size,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            #[BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.layers.dropout(char_embeddings,
                                                rate=cfg.output_keep_probability,
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
                                                    cfg.char_embedding_size],
                                             name="reduce_dimension_1")

                tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

                # word_lengths = get_sequence_length_old(char_embeddings) TODO working
                word_lengths = get_sequence_length(char_ids_reshaped)

                tf.logging.info('word_lengths =====> {}'.format(word_lengths))

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(cfg.char_level_lstm_hidden_size,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(cfg.char_level_lstm_hidden_size,
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
                                                  cfg.char_level_lstm_hidden_size])

                tf.logging.info('encoded_words =====> {}'.format(encoded_words))

        with  tf.variable_scope("word_level_lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(cfg.word_level_lstm_hidden_size,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(cfg.word_level_lstm_hidden_size,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=cfg.output_keep_probability)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=cfg.output_keep_probability)

                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  cfg.num_word_lstm_layers,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  cfg.num_word_lstm_layers,
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

            encoded_doc = tf.layers.dropout(encoded_doc, rate=cfg.output_keep_probability, seed=42,
                                            training=mode == tf.estimator.ModeKeys.TRAIN)

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

        with tf.name_scope("hidden_layer"):
            encoded_doc_shape = tf.shape(encoded_doc)
            BATCH_SIZE = encoded_doc_shape[0]

            size = ((cfg.word_level_lstm_hidden_size * 2) + \
                   (cfg.char_level_lstm_hidden_size * 2)) * \
                    self.MAX_DOC_LENGTH

            encoded_doc = tf.reshape(encoded_doc, shape=[BATCH_SIZE, size])

            tf.logging.info('encoded_doc: =====> {}'.format(encoded_doc))

            combined_logits = tf.layers.dense(inputs=encoded_doc,
                                     units=28*28*1,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
                                     activation=tf.nn.relu)

            combined_logits = tf.reshape(combined_logits, shape=[cfg.batch_size, 28,28,1])

            tf.logging.info('combined_logits: =====> {}'.format(combined_logits))



        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(combined_logits, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')

            tf.logging.info('conv1: =====> {}'.format(conv1))

            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

            # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

            # DigitCaps layer, return [batch_size, self.NUM_CLASSES, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.NUM_CLASSES, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

            # Decoder structure in Fig. 2
            # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, self.NUM_CLASSES, 16, 1] => [batch_size, self.NUM_CLASSES, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, dim=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, self.NUM_CLASSES, 1, 1]

            # b). pick out the index of max softmax val of the self.NUM_CLASSES caps
            # [batch_size, self.NUM_CLASSES, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))

            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, self.NUM_CLASSES, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(labels, (-1, self.NUM_CLASSES, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

                # 2. Reconstructe the MNIST images with 3 FC layers
                # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

        with tf.variable_scope("loss"):
            # 1. The margin loss

            # [batch_size, self.NUM_CLASSES, 1, 1]
            # max_l = max(0, m_plus-||v_c||)^2
            max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
            # max_r = max(0, ||v_c||-m_minus)^2
            max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
            assert max_l.get_shape() == [cfg.batch_size, self.NUM_CLASSES, 1, 1]

            # reshape: [batch_size, self.NUM_CLASSES, 1, 1] => [batch_size, self.NUM_CLASSES]
            max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
            max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

            # calc T_c: [batch_size, self.NUM_CLASSES]
            # T_c = Y, is my understanding correct? Try it.
            T_c = labels
            # [batch_size, self.NUM_CLASSES], element-wise multiply
            L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

            # 2. The reconstruction loss
            orgin = tf.reshape(combined_logits, shape=(cfg.batch_size, -1))
            squared = tf.square(self.decoded - orgin)
            self.reconstruction_err = tf.reduce_mean(squared)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            global_step = tf.contrib.framework.get_global_step()
            learning_rate = cfg.learning_rate
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