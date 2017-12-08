import os
import pandas as pd
from tqdm import tqdm
from tensorflow.python.platform import gfile
import tensorflow as tf
import ntpath

# import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

from utils.spacy import tokenize

from utils.colorful_logger import *

def size_mb(docs):
    # Each char is a byte and divide it by 100000
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


class TextDataFrame():
    def __init__(self,
                 train_df_path: str,
                 text_col: str,
                 category_col: str,
                 test_df_path: str='',
                 max_doc_legth=100,
                 max_word_length=15,
                 model_name="model_name" ):
        '''
        A simple class to maintain train/validation/test datasets, backed by Pandas Dataframe
        
        1. Load the train/val/test files into Pandas Dataframe
        2. Prepare the dataset
            - Prepare LabelTokenizer and LabelTokenizer
            - If test data is not given, train data set is split into train/validation/test set
            - If test data is given, train data set is split into train/validation set
        3. Tokenize the sentence
        4. Save the processed files
        5. Prepare features for trianing/validation/testing
            - Raw text data [NUM_OF_EXAMPLES,]
            - Text word IDs [NUM_OF_EXAMPLES, MAX_DOC_LENGTH]
            - Text word char IDs [NUM_OF_EXAMPLES, MAX_DOC_LENGTH, MAX_CHAR_LENGTH]
            - Indexed category labels [NUM_OF_EXAMPLES,]
            - One hot encoded labels [NUM_OF_EXAMPLES, NUM_CLASSES]
        
        :param train_df_path: Path of the train file that can be opened by Pandas as Dataframe
        :param test_df_path: Path of the train file that can be opened by Pandas as Dataframe
        :param text_col: Text column Name
        :param category_col: Expected category column name
        '''

        self.train_df_path: str = train_df_path
        self.test_df_path: str = test_df_path
        self.text_col = text_col
        self.category_col = category_col

        self.MAX_DOC_LEGTH = max_doc_legth
        self.MAX_WORD_LENGTH = max_word_length

        self.train_df, self.val_df, self.test_df = None, None, None

        self.le = LabelEncoder()
        self.label_binarizer = LabelBinarizer()

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        model_name = model_name+"/"
        self.words_vocab_file = model_name + "words_vocab.tsv"

        # Tokenize the sentences and store them
        if not os.path.exists(model_name+"train_processed.csv") or \
                not os.path.exists(model_name+"test_processed.csv") or \
                not os.path.exists(model_name+"val_processed.csv"):

            self._prepare_data()

            self.train_df = tokenize(self.train_df, text_col)
            self.val_df = tokenize(self.val_df, text_col)
            self.test_df = tokenize(self.test_df, text_col)

            self.train_df.to_csv(model_name+"train_processed.csv")
            self.val_df.to_csv(model_name + "val_processed.csv")
            self.test_df.to_csv(model_name + "test_processed.csv")

        else:
            print("Loading processed files...")
            self.train_df = pd.read_csv(model_name+"train_processed.csv")
            self.val_df = pd.read_csv(model_name + "val_processed.csv")
            self.test_df = pd.read_csv(model_name + "test_processed.csv")

            print('Fitting LabelEncoder and LabelBinarizer on processed utils...')
            self.le.fit(list(self.train_df[self.category_col]))
            self.label_binarizer.fit(list(self.train_df[self.category_col]))
            print('Done!')

        print("Preparing vocab file...")
        self.vocab_count, self.vocab = naive_vocab_creater(self.train_df[text_col].tolist(),
                                              self.words_vocab_file)

        self.word2id = {word: id for id, word in enumerate(self.vocab)}
        self.id2word = {id:word for id, word in enumerate(self.vocab)}

        self.char_vocab = get_char_vocab(self.vocab)

        self.char2id = {word: id for id, word in enumerate(self.char_vocab)}
        self.id2char = {id: word for id, word in enumerate(self.char_vocab)}

        self.WORD_VOCAB_SIZE = len(self.word2id)
        self.CHAR_VOCAB_SIZE = len(self.char2id)

        self.num_train_samples = self.train_df.shape[0]


    # TODO find an easy way???
    def _get_train_val_test_split(self, df):
        print('Splitting the utils set(stratified sampling)...')

        def train_validate_test_split(df, train_percent=.7, validate_percent=.2, seed=42):
            np.random.seed(seed)
            perm = np.random.permutation(df.index)
            m = len(df)
            train_end = int(train_percent * m)
            validate_end = int(validate_percent * m) + train_end
            train = df.loc[perm[:train_end]]
            validate = df.loc[perm[train_end:validate_end]]
            test = df.loc[perm[validate_end:]]
            return train, validate, test

        # Make list of sampled dataframe for each category
        dfs = [train_validate_test_split(df[df[self.category_col] == label]) for label in self.le.classes_]

        # Now the values are grouped to form a Dataframe
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for train_df, val_df, test_df in dfs:
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)

        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)
        test_df = pd.concat(test_dfs)

        # Shuffle the utils
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        print('Done!')

        return train_df, val_df, test_df

    def _get_train_val_split(self, df):
        print('Splitting the utils set(stratified sampling)...')

        def train_validate_test_split(df, train_percent=.8, seed=42):
            np.random.seed(seed)
            perm = np.random.permutation(df.index)
            m = len(df)
            train_end = int(train_percent * m)
            train = df.loc[perm[:train_end]]
            validate = df.loc[perm[train_end:]]
            return train, validate

        # Make list of sampled dataframe for each category
        dfs = [train_validate_test_split(df[df[self.category_col] == label]) for label in self.le.classes_]

        # Now the values are grouped to form a Dataframe
        train_dfs = []
        val_dfs = []
        for train_df, val_df in dfs:
            train_dfs.append(train_df)
            val_dfs.append(val_df)

        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)

        # Shuffle the utils
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)

        print('Done!')

        return train_df, val_df


    def _prepare_data(self):

        # If there is no test data, split the train data into train and validation
        if not os.path.exists(self.test_df_path):
            self.train_df = self.get_train_df()

            print('Fitting LabelEncoder and LabelBinarizer...')
            self.le.fit(list(self.train_df[self.category_col]))
            self.label_binarizer.fit(list(self.train_df[self.category_col]))
            print('Done!')

            self.train_df, self.val_df, self.test_df = self._get_train_val_test_split(self.train_df)
        else:
            self.train_df = self.get_train_df()

            print('Fitting LabelEncoder and LabelBinarizer...')
            self.le.fit(list(self.train_df[self.category_col]))
            self.label_binarizer.fit(list(self.train_df[self.category_col]))
            print('Done!')
            
            self.train_df, self.val_df = self._get_train_val_split(self.train_df)

            if self.test_df_path.endswith('.csv'):
                self.test_df = pd.read_csv(self.test_df_path)
            elif self.test_df_path.endswith('.json'):
                self.test_df = pd.read_json(self.test_df_path)
            else:
                self.test_df = pd.read_pickle(self.test_df_path)

    # Internal data set
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def _get_text_data(self, what='train_df'):
        if what == 'train_df':
            df = self.train_df[self.text_col]
        if what == 'test_df':
            df = self.test_df[self.text_col]
        if what == 'val_df':
            df = self.val_df[self.text_col]

        # return df.map(custom_replace).as_matrix()
        return df.as_matrix()

    def _get_target_label(self, df):
        print('Labels and their document counts based on', end=' ')
        print(df.groupby(self.category_col)[self.category_col].count())

        return self.le.transform(df[self.category_col])

    def _get_one_hot_target_label(self, df):
        print('Labels and their document counts based on', end=' ')
        print(df.groupby(self.category_col)[self.category_col].count())

        return self.label_binarizer.transform(df[self.category_col].as_matrix())

    def _get_text_word_ids(self, df):
        PAD_WORD_ID = 0
        UNKNOWN_WORD_ID = 1
        lines = df[self.text_col].as_matrix()
        tokenized_lines_ids = [[self.word2id.get(word, UNKNOWN_WORD_ID) for word in line.split(" ")] for line in lines]
        tokenized_lines_ids_padded, tokenized_lines_ids_length = pad_sequences(tokenized_lines_ids,
                                                                               nlevels=1,
                                                                               pad_tok=PAD_WORD_ID,
                                                                               max_doc_legth=self.MAX_DOC_LEGTH,
                                                                               max_word_length=self.MAX_WORD_LENGTH)
        return np.array(tokenized_lines_ids_padded)

    def _get_text_word_char_ids(self, df):
        PAD_CHAR_ID = 0
        UNKNOWN_CHAR_ID = 1
        lines = df[self.text_col].as_matrix()
        tokenized_words_id = []
        lines_char_ids = []
        char_ids = []

        for line in lines:
            for word in line.split(" "):
                for char in word:
                    char_ids.append(self.char2id.get(char, UNKNOWN_CHAR_ID))
                lines_char_ids.append(char_ids)
                char_ids = []
            tokenized_words_id.append(lines_char_ids)
            lines_char_ids = []

        tokenized_words_id_padded, tokenized_words_id_length = pad_sequences(tokenized_words_id,
                                                                             pad_tok=PAD_CHAR_ID,
                                                                            nlevels=2,
                                                                             max_word_length=self.MAX_WORD_LENGTH,
                                                                             max_doc_legth=self.MAX_DOC_LEGTH)

        return np.array(tokenized_words_id_padded)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_train_df(self):
        if self.train_df_path.endswith("csv"):
            full_df = pd.read_csv(self.train_df_path)
        elif self.train_df_path.endswith("json"):
            full_df = pd.read_json(self.train_df_path)
        else:
            full_df = pd.read_pickle(self.train_df_path)
        return full_df

    def get_test_df(self):
        return self.test_df

    def get_train_text_data(self):
        # return self.train_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('train_df')
        print("Size of train utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_val_text_data(self):
        # return self.val_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('val_df')
        print("Size of validation utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_test_text_data(self):
        # return self.test_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('test_df')
        print("Size of test utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_train_text_word_ids(self):
        docs = self._get_text_word_ids(self.train_df)
        # print("Size of train utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_val_text_word_ids(self):
        docs = self._get_text_word_ids(self.val_df)
        # print("Size of train utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_test_text_word_ids(self):
        docs = self._get_text_word_ids(self.test_df)
        # print("Size of train utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_train_text_word_char_ids(self):
        ret = self._get_text_word_char_ids(self.train_df)
        return ret

    def get_val_text_word_char_ids(self):
        ret = self._get_text_word_char_ids(self.val_df)
        return ret

    def get_test_text_word_char_ids(self):
        ret = self._get_text_word_char_ids(self.test_df)
        return ret

    def get_train_label(self):
        return self._get_target_label(self.train_df)

    def get_val_label(self):
        return self._get_target_label(self.val_df)

    def get_test_label(self):
        return self._get_target_label(self.test_df)

    def get_train_one_hot_label(self):
        return self._get_one_hot_target_label(self.train_df)

    def get_val_one_hot_label(self):
        return self._get_one_hot_target_label(self.val_df)

    def get_test_one_hot_label(self):
        return self._get_one_hot_target_label(self.test_df)

#==============================================================================

def naive_vocab_creater(lines, out_file_name):
    final_vocab = ["<PAD>", "<UNK>"]
    vocab = [word for line in lines for word in line.split(" ")]
    vocab = set(vocab)

    try:
        vocab.remove("<UNK>")
    except:
        print("No <UNK> token found")

    vocab = list(vocab)
    final_vocab.extend(vocab)

    print_warn(out_file_name)

    # Create a file and store the words
    with gfile.Open(out_file_name, 'wb') as f:
        for word in final_vocab:
            f.write("{}\n".format(word))

    return len(final_vocab), final_vocab

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
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
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
        max_length  = max_doc_legth
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)
        #breaking the code to pad the string instead on its ids

        # print_info(sequence_length)
    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq))
        #                        for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in tqdm(sequences):
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_word_length)
            sequence_padded += [sp]
            sequence_length += [sl]

        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*max_word_length,
                                            max_doc_legth)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_doc_legth)

    return sequence_padded, sequence_length

def tf_vocab_processor(lines, out_file_name, max_doc_length=1000, min_frequency=0):
    # Create vocabulary
    # min_frequency -> consider a word if and only it repeats for fiven count
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length,
                                                                         min_frequency=min_frequency)
    vocab_processor.fit(lines)

    # Create a file and store the words
    with gfile.Open(out_file_name, 'wb') as f:
        for word, index in vocab_processor.vocabulary_._mapping.items():
            f.write("{}\n".format(word))

    return len(vocab_processor.vocabulary_) + 1 #<UNK>


def get_vocab(df: pd.DataFrame, text_col: str):
    '''

    :param df: Pandas DataFrame
    :param text_col: Text Column
    :return:
    '''
    vocab = set()

    row_wise_tokens = df[text_col].str.split(" ").values
    try:
        for row in row_wise_tokens:
            for token in row:
                vocab.add(token)
    except:
        print(df)

    return vocab

def get_vocab_from_csvs(csv_files_path: str, text_col: str,
                   unkown_token = '<UNK>',
                   custom_tokens = ("<START>", "<END>")):
    '''

    :param csv_files_path: Path to CSV files, that can read by Pandas
    :param text_col:
    :return:
    '''
    vocab = set()
    for file in tqdm(os.listdir(csv_files_path)):
        file = os.path.join(csv_files_path, file)
        if file.endswith('.csv') and (os.path.getsize(file) > 0):
            df = pd.read_csv(file).fillna(unkown_token)
            vocab = vocab.union(get_vocab(df, text_col))

    if custom_tokens:
        vocab = vocab.union(custom_tokens)

    return sorted(vocab)

def get_char_vocab(words_vocab):
    '''

    :param words_vocab: List of words
    :return:
    '''
    chars = set()
    for word in words_vocab:
        for char in word:
            chars.add(char)
    return sorted(chars)

def get_char_vocab_from_df(df: pd.DataFrame, text_col):
    '''
    Function that takes dataframe and a text column name, to generate
    char vocab for later use
    Eg: char to char_id
    :param df: Pandas DataFrame
    :param text_col:
    :return:
    '''
    return get_char_vocab(get_vocab(df, text_col))

def build_vocab(csv_file_path: str, text_col: str, out_file_path: str, custom_tokens=None):
    words_vocab = get_vocab_from_csvs(csv_file_path, text_col, custom_tokens=None)
    chars_vocab = get_char_vocab(words_vocab)

    vocab_to_tsv(words_vocab, out_file_path+"/" + text_col+ "_" + "words_vocab.tsv")
    vocab_to_tsv(chars_vocab, out_file_path+"/" + text_col+ "_" + "chars_vocab.tsv")

