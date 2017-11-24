from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

# Parameters for downloading utils
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
FILE_NAME = 'text8.zip'

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
    	pass

def download(file_name, expected_bytes, download_path):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = download_path + "/" + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    else:
        os.makedirs(download_path)
    print("Downloading the data...")
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path

def read_data(file_path):
    """ Read utils into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    word_2_id = dict()
    count = [('<UNK>', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    for word, _ in count:
        word_2_id[word] = index
        index += 1
    id_2_word = dict(zip(word_2_id.values(), word_2_id.keys()))
    return word_2_id, id_2_word

def generate_sample(words, context_window_size):
    """ Form training pairs according to the skip-gram model. """

    center_words = []
    target_words = []

    for index, center in enumerate(words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in words[max(0, index - context): index]:
            center_words.append(center)
            target_words.append(target)
        # get a random target after the center wrod
        for target in words[index + 1: index + context + 1]:
            center_words.append(center)
            target_words.append(target)

    return center_words, target_words

#--------------------------------------------------------------------------------------
#
# def generate_sample1(index_words, context_window_size):
#     """ Form training pairs according to the skip-gram model. """
#     for index, center in enumerate(index_words):
#         context = random.randint(1, context_window_size)
#         # get a random target before the center word
#         for target in index_words[max(0, index - context): index]:
#             yield center, target
#         # get a random target after the center wrod
#         for target in index_words[index + 1: index + context + 1]:
#             yield center, target
#
# def convert_words_to_index(words, dictionary):
#     """ Replace each word in the dataset with its index in the dictionary """
#     return [dictionary[word] if word in dictionary else 0 for word in words]
#
#
#
# def get_batch(iterator, batch_size):
#     """ Group a numerical stream into batches and yield them as Numpy arrays. """
#     while True:
#         center_batch = np.zeros(batch_size, dtype=np.int32)
#         target_batch = np.zeros([batch_size, 1])
#         for index in range(batch_size):
#             center_batch[index], target_batch[index] = next(iterator)
#         yield center_batch, target_batch
#
# def process_data(vocab_size, batch_size, skip_window):
#     file_path = download(FILE_NAME, EXPECTED_BYTES, "tmp/")
#     words = read_data(file_path)
#     dictionary, _ = build_vocab(words, vocab_size)
#     index_words = convert_words_to_index(words, dictionary)
#     del words # to save memory
#     single_gen = generate_sample(index_words, skip_window)
#     return get_batch(single_gen, batch_size)
#
# def get_index_vocab(vocab_size):
#     file_path = download(FILE_NAME, EXPECTED_BYTES, "tmp/")
#     words = read_data(file_path)
#     return build_vocab(words, vocab_size)