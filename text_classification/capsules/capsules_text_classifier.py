import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys
from capsules.capsules_config import cfg


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

class CapsulesTextClassifier(tf.estimator.Estimator):
    def __init__(self,
                 config,
                 vocab_size,
                 char_vocab_size):
        super(CapsulesTextClassifier, self).__init__(
            model_fn=self._model_fn,
            model_dir=cfg.MODEL_DIR,
            config=run_config)

        self.WORD_VOCAB_SIZE = vocab_size
        self.CHAR_VOCAB_SIZE = char_vocab_size

    def _model_fn(self, features, labels, mode, params):
        '''

        :param features: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param labels: TF Placeholder of type String of shape [BATCH_SIZE, 1]
        :param mode: ModeKeys
        :param params:
        :return:
        '''

        is_training = mode == ModeKeys.TRAIN

        # [BATCH_SIZE, 1]
        token_ids = features['word_ids']

        # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH]
        char_ids = features['char_ids']

        print("\n\n\n\n\n")
        tf.logging.info('token_ids: =======> {}'.format(token_ids))
        tf.logging.info('char_ids: =======> {}'.format(char_ids))
        tf.logging.info('labels: =======> {}'.format(labels))

        s = tf.shape(char_ids)

        #remove pad words
        char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2])) #20 -> char dim

        with tf.name_scope("word-embed-layer"):
            # layer to take the words and convert them into vectors (embeddings)
            # This creates embeddings matrix of [VOCAB_SIZE, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into
            # [BATCH_SIZE, MAX_SEQ_LENGTH] --->  [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE].
            word_embeddings = tf.contrib.layers.embed_sequence(token_ids,
                                                               vocab_size=cfg.VOCAB_SIZE,
                                                               embed_dim=cfg.WORD_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            word_embeddings = tf.layers.dropout(word_embeddings,
                                                rate=cfg.KEEP_PROP,
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
                                                               embed_dim=cfg.CHAR_EMBEDDING_SIZE,
                                                               initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))

            #[BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
            char_embeddings = tf.layers.dropout(char_embeddings,
                                                rate=cfg.KEEP_PROP,
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
                                                    cfg.CHAR_EMBEDDING_SIZE],
                                             name="reduce_dimension_1")

                tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

                # word_lengths = get_sequence_length_old(char_embeddings) TODO working
                word_lengths = get_sequence_length(char_ids_reshaped)

                tf.logging.info('word_lengths =====> {}'.format(word_lengths))

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(cfg.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(cfg.CHAR_LEVEL_LSTM_HIDDEN_SIZE,
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
                                                  cfg.CHAR_LEVEL_LSTM_HIDDEN_SIZE])

                tf.logging.info('encoded_words =====> {}'.format(encoded_words))

        with  tf.name_scope("word_level_lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(cfg.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(cfg.WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=cfg.OUTPUT_KEEP_PROBABILITY)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=cfg.OUTPUT_KEEP_PROBABILITY)

                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  cfg.NUM_WORD_LSTM_LAYERS,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  cfg.NUM_WORD_LSTM_LAYERS,
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