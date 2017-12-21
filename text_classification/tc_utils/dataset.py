import os
import sys
sys.path.append("../../")
import pandas as pd
from tqdm import tqdm
from tensorflow.python.platform import gfile
import tensorflow as tf
# import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

from sarvam.colorful_logger import *
from sarvam.nlp.spacy import tokenize

def size_mb(docs):
    # Each char is a byte and divide it by 100000
    return sum(len(str(s).encode('utf-8')) for s in docs) / 1e6


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
        for seq in tqdm(sequences, desc="pad_sequences"):
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
    for file in tqdm(os.listdir(csv_files_path), desc="get_vocab_from_csvs"):
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



class TextDataFrame():
    def __init__(self,
                 train_file_path,
                 test_file_path,
                 text_col,
                 category_col,
                 category_cols,
                 max_doc_legth,
                 max_word_length,
                 is_multi_label,
                 is_tokenize,
                 dataset_name="dataset"):
        '''
        A simple class to maintain train/validation/test datasets, backed by Pandas Dataframe
        
        1. Load the train and test files into Pandas Dataframe
        2. Prepare the dataset
            - Prepare LabelTokenizer and LabelTokenizer
            - If test data is not given, train data set is split into train/validation/test set
            - If test data is given, train data set is split into train/validation set
        3. Tokenize the sentence with spaCy model "en_core_web_lg"
        4. Save the processed files
        5. Prepare features for trianing/validation/testing
            - Account for category column
                - User LabelEncoder and LabelBinarizer to get index and on hot encoded labels
            - Account for category columns and does not contain multi labels 
                - Convert category columns ---> inffered category columns and use above method down the line
            - Account for category columns and does contain multi labels
                - No stratified sampling possible based on labels
                - Go with random sampling
                - Use actual category colums for one-hot coded labels
            - Raw text data [NUM_OF_EXAMPLES,]
            - Text word IDs [NUM_OF_EXAMPLES, MAX_DOC_LENGTH]
            - Text word char IDs [NUM_OF_EXAMPLES, MAX_DOC_LENGTH, MAX_CHAR_LENGTH]
            - Indexed category labels [NUM_OF_EXAMPLES,]
            - One hot encoded labels [NUM_OF_EXAMPLES, NUM_CLASSES]
        
        :param train_file_path: Path of the train file that can be opened by Pandas as Dataframe
        :param test_file_path: Path of the test file that can be opened by Pandas as Dataframe
        :param text_col: Text column Name
        :param category_col: Expected category column name
        :param category_cols: One hot encoded column names for categories
        :param max_doc_legth: Maximum length of the document to consider
        :param max_word_length:  Maximum length of the word to consider
        '''

        self.train_df_path: str = train_file_path
        self.test_df_path: str = test_file_path
        self.text_col = text_col

        if category_col == None:
            self.category_col = "category_inffered"
        else:
            self.category_col = category_col

        self.category_cols = category_cols

        self.is_multi_label = is_multi_label
        self.is_tokenize = is_tokenize

        self.MAX_DOC_LEGTH = max_doc_legth
        self.MAX_WORD_LENGTH = max_word_length

        self.train_df, self.val_df, self.test_df = None, None, None

        self.le = LabelEncoder()
        self.label_binarizer = LabelBinarizer()

        if not os.path.exists(dataset_name):
            os.makedirs(dataset_name)

        dataset_name = dataset_name + "/"
        self.words_vocab_file = dataset_name + "words_vocab.tsv"

        # Tokenize the sentences and store them
        if not os.path.exists(dataset_name+ "train_processed.csv") or \
                not os.path.exists(dataset_name+ "test_processed.csv") or \
                not os.path.exists(dataset_name+ "val_processed.csv"):

            self._prepare_data()

            if self.is_tokenize:
                print("Tokenizing...")
                self.train_df = tokenize(self.train_df, text_col)
                self.train_df.to_csv(dataset_name + "train_processed.csv")
                self.val_df = tokenize(self.val_df, text_col)
                self.val_df.to_csv(dataset_name + "val_processed.csv")
                self.test_df = tokenize(self.test_df, text_col)

            self.train_df.to_csv(dataset_name + "train_processed.csv")
            self.val_df.to_csv(dataset_name + "val_processed.csv")
            self.test_df.to_csv(dataset_name + "test_processed.csv")

        else:
            print("Loading processed files...")
            self.train_df = pd.read_csv(dataset_name + "train_processed.csv")
            self.val_df = pd.read_csv(dataset_name + "val_processed.csv")
            self.test_df = pd.read_csv(dataset_name + "test_processed.csv")

            if not self.is_multi_label:
                print('Fitting LabelEncoder and LabelBinarizer on processed audio_utils...')
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


    def _get_train_val_test_split(self, df):
        print('Splitting the audio_utils set(stratified sampling)...')

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

        # Shuffle the audio_utils
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        print('Done!')

        return train_df, val_df, test_df

    def _get_train_val_split(self, df):
        print('Splitting the train set into trai and vaidation (stratified sampling)...')

        def train_validate_test_split(df, train_percent=.95, seed=42):
            np.random.seed(seed)
            perm = np.random.permutation(df.index)
            m = len(df)
            train_end = int(train_percent * m)
            train = df.loc[perm[:train_end]]
            validate = df.loc[perm[train_end:]]
            return train, validate

        # Make list of sampled dataframe for each category
        dfs = [train_validate_test_split(df[df[self.category_col] == label])
               for label in tqdm(self.le.classes_, desc="_get_train_val_split")]

        # Now the values are grouped to form a Dataframe
        train_dfs = []
        val_dfs = []
        for train_df, val_df in tqdm(dfs):
            train_dfs.append(train_df)
            val_dfs.append(val_df)

        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)

        # Shuffle the audio_utils
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)

        print('Done splitting!')

        return train_df, val_df

    def one_hot_2_categories(self, df):
        '''
        Function to convert one hpt encoded labels to category nams based on the column names.
        This extra step is done in order to use label encoders and use stratified sampling
        :param df: 
        :return: 
        '''
        if self.category_cols and not self.is_multi_label:
            category_map = dict(enumerate(self.category_cols))
            labels = df[self.category_cols].as_matrix().argmax(axis=1)
            labels = list(map(lambda label: category_map[label], labels))
            df[self.category_col] = labels
        else:
            df = df
        return df


    def _prepare_data(self):

        if self.is_multi_label:
            if not os.path.exists(self.test_df_path):
                print_info(self.test_df_path)
                raise NotImplementedError("Spend some time to implement this!")
            else:
                self.train_data = self.get_train_data()
                if self.test_df_path.endswith('.csv'):
                    self.test_df = pd.read_csv(self.test_df_path)
                elif self.test_df_path.endswith('.json'):
                    self.test_df = pd.read_json(self.test_df_path)
                else:
                    self.test_df = pd.read_pickle(self.test_df_path)

                mask = np.random.rand(len(self.train_data)) < 0.95
                #TODO train_test_split(df, test_size=0.05)
                self.train_df, self.val_df = self.train_data[mask], self.train_data[~mask]
        else:
            # If there is no test data, split the train data into train and validation
            if not os.path.exists(self.test_df_path):
                self.train_df = self.get_train_data()

                print('Fitting LabelEncoder and LabelBinarizer...')
                self.le.fit(list(self.train_df[self.category_col]))
                self.label_binarizer.fit(list(self.train_df[self.category_col]))
                print('Done label encoding !')


                self.train_df, self.val_df, self.test_df = self._get_train_val_test_split(self.train_df)
            else:
                self.train_df = self.get_train_data()

                self.train_df = self.one_hot_2_categories(self.train_df)

                print('Fitting LabelEncoder and LabelBinarizer...')
                self.le.fit(list(self.train_df[self.category_col]))
                self.label_binarizer.fit(list(self.train_df[self.category_col]))
                print('Done label encoding !')

                self.train_df, self.val_df = self._get_train_val_split(self.train_df)

                if self.test_df_path.endswith('.csv'):
                    self.test_df = pd.read_csv(self.test_df_path)
                elif self.test_df_path.endswith('.json'):
                    self.test_df = pd.read_json(self.test_df_path)
                else:
                    self.test_df = pd.read_pickle(self.test_df_path)


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

        if not self.is_multi_label:
            print('Labels and their document counts based on', end=' ')
            print(df.groupby(self.category_col)[self.category_col].count())
        if self.category_cols:
            if self.is_multi_label:
                print_error("Not possible with multi label setup!")
                return None
            else:
                return df[self.category_cols].as_matrix().argmax(axis=1)
        else:
            return self.le.transform(df[self.category_col])

    def _get_one_hot_target_label(self, df):
        if not self.is_multi_label:
            print('Labels and their document counts based on', end=' ')
            print(df.groupby(self.category_col)[self.category_col].count())
        if self.category_cols:
            return df[self.category_cols].as_matrix()
        else:
            return self.label_binarizer.transform(df[self.category_col].as_matrix())

    def _get_text_word_ids(self, df):
        PAD_WORD_ID = 0
        UNKNOWN_WORD_ID = 1
        lines = df[self.text_col].as_matrix()
        tokenized_lines_ids = [[self.word2id.get(word, UNKNOWN_WORD_ID) for word in str(line).split(" ")] for line in tqdm(lines)]
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

        for line in tqdm(lines, desc="_get_text_word_char_ids:"):
            for word in str(line).split(" "):
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

    def get_train_data(self):
        '''
        Use this function to get the full training data.
        Reads the train file every time this is called. 
        :return: Pandas Dataframe of the training file
        '''
        if self.train_df_path.endswith("csv"):
            full_df = pd.read_csv(self.train_df_path)
        elif self.train_df_path.endswith("json"):
            full_df = pd.read_json(self.train_df_path)
        else:
            full_df = pd.read_pickle(self.train_df_path)

        full_df = self.one_hot_2_categories(full_df)
        return full_df

    def get_test_df(self):
        '''
        Use this function to get the full test data
        :return: Pandas Dataframe of the test file
        '''
        return self.test_df

    def get_train_text_data(self):
        # return self.train_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('train_df')
        print("Size of train data: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_val_text_data(self):
        # return self.val_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('val_df')
        print("Size of validation data: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_test_text_data(self):
        # return self.test_df[self.text_col].map(lambda x: self.replace(x)).as_matrix()
        docs = self._get_text_data('test_df')
        print("Size of test data: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_train_text_word_ids(self):
        docs = self._get_text_word_ids(self.train_df)
        # print("Size of train audio_utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_val_text_word_ids(self):
        docs = self._get_text_word_ids(self.val_df)
        # print("Size of train audio_utils: %0.3fMB" % (size_mb(docs)))
        return docs

    def get_test_text_word_ids(self):
        docs = self._get_text_word_ids(self.test_df)
        # print("Size of train audio_utils: %0.3fMB" % (size_mb(docs)))
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


