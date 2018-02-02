import sys
import os
sys.path.append("../")

from tqdm import tqdm
import tensorflow as tf
import numpy as np
from nlp.text_classification.dataset.data_iterator import DataIterator
from nlp.text_classification.dataset.feature_types import TextIdsFeature
from nlp.text_classification.tc_utils.tf_hooks.data_initializers import IteratorInitializerHook
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.python.platform import gfile
from sarvam.config.global_constants import *

from sarvam.helpers.print_helper import *
import spacy
import threading

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels, max_doc_legth, max_word_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max_doc_legth
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)
        # breaking the code to pad the string instead on its ids

        # print_info(sequence_length)
    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq))
        #                        for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in tqdm(sequences, desc="pad_sequences"):
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_word_length)
            sequence_padded += [sp]
            sequence_length += [sl]

        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_word_length,
                                            max_doc_legth)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_doc_legth)

    return sequence_padded, sequence_length

class LazyGeneratorTextIds(DataIterator):
    def __init__(self, batch_size, dataframe, num_epochs):
        DataIterator.__init__(self)
        self.feature_type = TextIdsFeature

        self._batch_size = batch_size
        self._dataframe = dataframe
        self._num_epochs = num_epochs
        self._nlp = spacy.load('en_core_web_md')

    def default_nlp_tokenizer(self, text, mode, out_file_name="/tmp/vocab.txt"):
        """
        On the file tokenization and vocab creator!
        !!!Lot Disk I/O involved use with care for big vocab list!!!
        :param text: 
        :param mode: 
        :param out_file_name: 
        :return: 
        """

        vocab = [PAD_WORD, UNKNOWN_WORD]
        # print_info(threading.current_thread().name)

        if os.path.exists(out_file_name):
            with open(out_file_name) as file:
                lines = file.readlines()
            lines = map(lambda line: line.strip(), lines)
            vocab = list(lines)

        final_vocab_dict = { word: i for i, word in enumerate(vocab)}
        current_last_index = len(final_vocab_dict) - 1

        # tokenized_text = [word.text for word in self._nlp(text)]
        tokenized_text = [word for word in text.split()]

        tokenized_text_id = []

        if mode == "train":
            for word in tokenized_text:
                if word in final_vocab_dict.keys():
                    tokenized_text_id.append(final_vocab_dict[word])
                else:
                    current_last_index = current_last_index + 1
                    final_vocab_dict[word] = current_last_index
                    tokenized_text_id.append(final_vocab_dict[word])

            with gfile.Open(out_file_name, 'wb') as f:
                for word in final_vocab_dict.keys():
                    f.write("{}\n".format(word))
        else:
            for word in tokenized_text:
                tokenized_text_id.append(final_vocab_dict.get(word, UNKNOWN_WORD_ID))

        tokenized_lines_ids_padded, tokenized_lines_ids_length = pad_sequences([tokenized_text_id],
                                                                               nlevels=1,
                                                                               pad_tok=PAD_WORD_ID,
                                                                               max_doc_legth=self._dataframe .MAX_DOC_LEGTH,
                                                                               max_word_length=self._dataframe .MAX_WORD_LENGTH)

        return np.array(tokenized_lines_ids_padded[0]).astype(int)

    def data_generator(self, text_data, one_hot_label, mode='train'):

        print_info("Total samples in current set: {}".format(len(text_data)))

        def generator():
            for i, text in tqdm(enumerate(text_data), desc=mode):
                ids = self.default_nlp_tokenizer(text, mode)

                try:
                    res = {self.feature_type.FEATURE_1 : ids,
                     self.feature_type.LABEL: np.array(one_hot_label[i]) }

                    yield res

                except Exception as err:
                    print_error(text)
                    print_error(one_hot_label[i])

        return generator


    def prepare_train_set(self):
        self.train_input_fn = generator_input_fn(
            x=self.data_generator(self._dataframe.get_train_text_data(),
                                  self._dataframe.get_train_one_hot_label(), 'train'),
            target_key=self.feature_type.LABEL,  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=self._num_epochs,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

    def prepare_val_set(self):
        self.val_input_fn = generator_input_fn(
            x=self.data_generator(self._dataframe.get_val_text_data(),
                                  self._dataframe.get_val_one_hot_label(), 'val'),
            target_key=self.feature_type.LABEL,
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

    def prepare_test_set(self):
        self.test_input_fn = generator_input_fn(
            x=self.data_generator(self._dataframe.get_test_text_data(),
                                  self._dataframe.get_test_one_hot_label(), 'test'),
            target_key=self.feature_type.LABEL,
            batch_size=self._batch_size,
            shuffle=False,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

