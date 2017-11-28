import codecs
import logging
from collections import Counter, defaultdict
import nltk

from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile

import tensorflow as tf

from six.moves import urllib
from tqdm import tqdm
from word_vec.utils.common import vocab_to_tsv

import glob
import numpy as np
import pickle

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except:
    	pass

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class GloveDataset():
    def __init__(self,
                 vocabulary_size=80000,
                 min_occurrences=5,
                 window_size=5,
                 name='GloveDataset',
                 text_dir=None):
        """
        1. Convert the lines in text file as tokens with NLP.
        2. Make pairs of  (left_context, word, right_context) for given margin
        3. Filter words based on their occurrences in the whole corpus
        4. Make a map of word -> index
        5. Construct cooccurrence_matrix with (word_index, context_index) : count 
        :param vocabulary_size: Number of word to be to vectorized
        :param min_occurrences: Minimum number of occurances for the word to be considered
        :param left_window_size: Left Margin Size
        :param right_window_size: Right Margin Size
        :param name: Name of the Model
        :param feature_type: 
        :param train_files: Any Text File Eg: Wiki Pages
        :param test_files: Any Text File Eg: Wiki Pages
        :param val_files:  Any Text File Eg: Wiki Pages
        """

        self.vocab = None
        self.word_to_id = None
        self.cooccurrence_matrix = None

        self.vocabulary_size = vocabulary_size
        self.min_occurrences = min_occurrences

        self.window_size = window_size
        self.left_window_size = window_size
        self.right_window_size = window_size
        self.text_dir = text_dir

        self.number_examples = 0

        # if train_files is None:
        #     self.download("tmp/")

    # def download(self, download_path):
    #     # Parameters for downloading utils
    #     expected_bytes = 31344016
    #     file_name = 'text8.zip'
    #     DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
    #
    #     """ Download the dataset text8 if it's not already downloaded """
    #     file_path = download_path + "/" + file_name
    #     if os.path.exists(file_path):
    #         print("Dataset ready")
    #         return file_path
    #     else:
    #         os.makedirs(download_path)
    #     print("Downloading the data...")
    #     file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    #     file_stat = os.stat(file_path)
    #     if file_stat.st_size == expected_bytes:
    #         print('Successfully downloaded the file', file_name)
    #     else:
    #         raise Exception('File ' + file_name +
    #                         ' might be corrupted. You should try downloading it with a browser.')
    #     return file_path
    #
    # def get_words(self, file_path="tmp/text8.zip"):
    #     """ Read utils into a list of tokens
    #     There should be 17,005,207 tokens
    #     """
    #     with zipfile.ZipFile(file_path) as f:
    #         words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    #         # tf.compat.as_str() converts the input into the string
    #     return words
    #
    # def get_lines(self, file_path="tmp/text8.zip"):
    #     """ Read utils into a list of tokens
    #     There should be 17,005,207 tokens
    #     """
    #     with zipfile.ZipFile(file_path) as f:
    #         lines = tf.compat.as_str(f.read(f.namelist()[0]))
    #         # tf.compat.as_str() converts the input into the string
    #     return lines

    #================================================================================================
    #   Simple Raw Skip Gram Word-Target Pair Generator
    #================================================================================================

    # def build_vocab(self, words):
    #     """ Build vocabulary of VOCAB_SIZE most frequent words """
    #
    #     word_2_id = dict()
    #     count = [('<UNK>', -1)]
    #     count.extend(Counter(words).most_common(self.vocabulary_size - 1))
    #     index = 0
    #     for word, _ in count:
    #         word_2_id[word] = index
    #         index += 1
    #     id_2_word = dict(zip(word_2_id.values(), word_2_id.keys()))
    #     return word_2_id, id_2_word

    def lines_to_words(self, list_of_tokenized_lines):

        if not isinstance(list_of_tokenized_lines, list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(list_of_tokenized_lines, type(list_of_tokenized_lines)))

        if not isinstance(list_of_tokenized_lines[0], list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(list_of_tokenized_lines, type(list_of_tokenized_lines)))

        words = []
        for list_tokens in list_of_tokenized_lines:
            words.extend(list_tokens)

        return words

    def generate_raw_skip_gram_samples(self, words):
        """ Form training pairs according to the skip-gram model. """

        center_words = []
        target_words = []

        for index, center in enumerate(words):
            # """ Form training pairs according to the skip-gram model. """
            context = self.window_size#random.randint(1, self.window_size)
            # get a random target before the center word
            for target in words[max(0, index - context): index]:
                if center in self.vocab and target in self.vocab:
                    center_words.append(center)
                    target_words.append(target)
            # get a random target after the center wrod
            for target in words[index + 1: index + context + 1]:
                if center in self.vocab and target in self.vocab:
                    center_words.append(center)
                    target_words.append(target)

        return center_words, target_words

    #================================================================================================
    #   Skip Gram Word-Count Pair Generator
    #================================================================================================


    def window(self, tokens, start_index, end_index):
        """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from tokens. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in tokens, this function will pad
        its return value with `NULL_WORD`.
        """
        last_index = len(tokens) + 1
        selected_tokens = tokens[max(start_index, 0):min(end_index, last_index) + 1]
        return selected_tokens

    def context_windows(self, tokens, left_window_size, right_window_size):
        """

        :param tokens: List of tokens
        :param left_window_size: 
        :param right_window_size: 
        :return: 
        """
        for i, word in enumerate(tokens):
            start_index = i - left_window_size
            end_index = i + right_window_size
            left_context = self.window(tokens, start_index, i - 1)
            right_context = self.window(tokens, i + 1, end_index)
            yield (left_context, word, right_context)

    def fit_to_corpus(self, list_of_tokenized_lines):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)

        # Page Number: 2
        # Let this be X
        # X_ij tabulates number of times word j (word[1]) occurs in context of word i (word[0])
        # Let X_i = Sum of X_ik for all k,  be the number of times any word appears in the context of word i.
        # Probability P_ij = P(j|i) = X_ij/X_i be the probability that word j appear in the context of word i.

        # Page Number: 3
        # The above argument suggests that the appropriate
        # starting point for word vector learning should
        # be with ratios of co-occurrence probabilities rather
        # than the probabilities themselves. Noting that the
        # ratio Pik /Pjk depends on three words i, j, and k,
        # the most general model takes the form,....

        # Let X_i = Sum of X_ik for all k,  be the number of times any word appears in the context of word i.
        # Probability P_ik = P(k|i) = X_ik/X_i be the probability that word k appear in the context of word i.

        # Let X_j = Sum of X_jk for all k,  be the number of times any word appears in the context of word j.
        # Probability P_jk = P(k|j) = X_jk/X_j be the probability that word k appear in the context of word j.
        for tokens in tqdm(list_of_tokenized_lines, desc="coocurr_mat: "):
            word_counts.update(tokens)
            for left_context, word_k, right_context in self.context_windows(tokens, self.left_window_size,
                                                                            self.right_window_size):
                # add (1 / distance from focal word) for this pair
                for i, context_word_i in enumerate(left_context[::-1]):
                    cooccurrence_counts[(word_k, context_word_i)] += 1 / (i + 1)  # 1 is added since index is from 0
                for j, context_word_j in enumerate(right_context):
                    cooccurrence_counts[(word_k, context_word_j)] += 1 / (j + 1)

        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in lines. Did you try to reuse a generator?")

        self.vocab = [word for word, count in word_counts.most_common(self.vocabulary_size)
                      if count >= self.min_occurrences]

        # self.word_to_id = {word: i for i, word in enumerate(self.vocab)}

        self.cooccurrence_matrix = {
            # (self.word_to_id[words[0]], self.word_to_id[words[1]]): count
            (word_tuple[0], word_tuple[1]): count
            for word_tuple, count in cooccurrence_counts.items()
            if word_tuple[0] in self.vocab and word_tuple[1] in self.vocab
        }


    def get_cooccurrence_features(self, list_of_tokenized_lines):
        """
        Read a dataset (basically a list of features) from
        a utils file.
        :param lines: List of str
            A list containing string representations of each
            line in the file.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the list.
        """
        if not isinstance(list_of_tokenized_lines, list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(list_of_tokenized_lines, type(list_of_tokenized_lines)))

        if not isinstance(list_of_tokenized_lines[0], list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(list_of_tokenized_lines, type(list_of_tokenized_lines)))

        logger.info("Fitting the lines to get Coocurrance Matrix")

        self.fit_to_corpus(list_of_tokenized_lines)

        logger.info('Extracting the features...')
        features = []
        for word_ids, counts in tqdm(self.cooccurrence_matrix.items()):
            i_indices = word_ids[0]
            j_indices = word_ids[1]
            feature = [i_indices, j_indices, counts]
            features.append(feature)
        print(features[:10])

        return features

    def read_from_files(self, text_dir):
        """
        Read a dataset (basically a list of features) from
        a utils file.
        :param file_names: str or List of str
                 The string filename from which to read the features, or a List of
            strings repesenting the files to pull features from. If a string
            is passed in, it is automatically converted to a single-element
            list.
        :param feature_class: Feature
            The Feature class to create from these lines.
        :return: text_dataset: TextDataset
            A new TextDataset with the features read from the file.
        """

        logger.info("Reading files from {} to a list of lines.".format(text_dir))
        file_names = []
        for filename in glob.iglob(text_dir + '**/**', recursive=True):
            if filename.split("/")[-1].startswith("wiki"):
                file_names.append(filename)

        print("List of files: {}".format(file_names))


        lines = [x.strip() for filename in file_names
                 for x in tqdm(codecs.open(filename, "r", "utf-8").readlines())]

        #[[tok1, tok2, tik3, ...], [toke4, tok2, tok5, ...], ...]
        list_of_tokenized_lines = [nltk.wordpunct_tokenize(line.lower()) for line in tqdm(lines, desc="tokenizing: ")]


        
        return list_of_tokenized_lines

    def prepare(self):
        data = {}

        list_of_tokenized_lines =  self.read_from_files(self.text_dir)

        cooccurrence_features = self.get_cooccurrence_features(list_of_tokenized_lines)
        cooccurrence_features = np.array(cooccurrence_features)
        self.number_examples = cooccurrence_features.shape[0]

        data["words"] = np.array(cooccurrence_features[:,0])
        data["targets"] = np.array(cooccurrence_features[:, 1])
        data["cooocurrence_count"] = np.array(cooccurrence_features[:, 2])

        del cooccurrence_features

        vocab_to_tsv(vocab_list=self.vocab, outfilename="tmp/vocab.tsv")

        words = self.lines_to_words(list_of_tokenized_lines)

        del list_of_tokenized_lines
        
        center_words, target_words = self.generate_raw_skip_gram_samples(words)

        data["center_words"] = center_words
        data["target_words"] = target_words

        with open("tmp/train_data.pickle", "wb") as file:
            pickle.dump(data, file)


        
    def embedding_for(self, word_str_or_id, embeddings):
        if isinstance(word_str_or_id, str):
            return embeddings[self.word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return embeddings[word_str_or_id]
            # TODO https://github.com/shashankg7/glove-tensorflow/blob/master/glove/utils.py