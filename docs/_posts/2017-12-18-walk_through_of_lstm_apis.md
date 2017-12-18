---
layout: post
title:  "LSTM Tensorflow APIs"
description: "A simple walk through on Tensorflow LSTM APIs"
excerpt: "A simple walk through on Tensorflow LSTM APIs"
date:   2017-12-17
mathjax: true
comments: true 
---

**Most of the code is self explanatory, if not comment **

# Text Data and LSTM 

Following are the high level data flow used in i-Tagger:
1. Preprocessing the data
2. Creating the vocab 
    - Word/Tokens vocab which includes **word -> id** & **id -> word**
    - Char vocab which includes **char -> id** & **id -> char**
3. Prepraing the features
    - Padding words/characters with custome function (or) 
    - Using Tensorflow APIs
4. Model
    - Word Embeddings
    - Word Level BiLSTM encoding ignoring the padded words
    - Character Embedding
    - Char Level BiLSTM encoding ignoring the padded characters


## You cant initialize graph components twice, if you encounter error, restart the notebook


```python
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import numpy as np
from tqdm import tqdm

tf.reset_default_graph()

```

    /home/mageswarand/anaconda3/envs/tensorflow1.0/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
      return f(*args, **kwds)


### Constants



```python


# Normally this takes the mean length of the words in the dataset documents
MAX_DOCUMENT_LENGTH = 7

# Padding word that is used when a document has less words than the calculated mean length of the words
PAD_WORD = '<PAD>'
PAD_WORD_ID = 0

PAD_CHAR = "<P>"
PAD_CHAR_ID = 0

UNKNOWN_WORD = "<UNKNOWN>"
UNKNOWN_WORD_ID = 1

UNKNOWN_CHAR = "<U>"
UNKNOWN_CHAR_ID = 1

EMBEDDING_SIZE = 3
WORD_EMBEDDING_SIZE = 3
CHAR_EMBEDDING_SIZE = 3
WORD_LEVEL_LSTM_HIDDEN_SIZE = 3
CHAR_LEVEL_LSTM_HIDDEN_SIZE = 3
```

### MISC


```python
value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value)

tf.reset_default_graph()
with tf.Session() :
    x = tf.get_variable('x', shape = [3, 4], initializer = init)
    x.initializer.run()
    print(x.eval())
```

    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]


There are places where "tf.get_variable" is used without a initializer, in which case the TF engine initializes it with default intializer


```python
tf.reset_default_graph()
with tf.Session() as sess:
    W = tf.get_variable("W", dtype=tf.float32, shape=[5,5])
    b = tf.get_variable("b", shape=[12],  dtype=tf.float32, initializer=tf.zeros_initializer())
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(b.eval())
```

    [[ 0.38060057  0.50141478  0.40662122  0.72093308 -0.29107267]
     [-0.2623179   0.30131912 -0.55878854  0.63164377 -0.46682838]
     [ 0.75094438  0.61107099 -0.59864926  0.54592323 -0.48808187]
     [-0.43495807 -0.13385278  0.19299984  0.512218    0.28124976]
     [ 0.29379117 -0.02249491  0.35649407  0.51151133 -0.64783204]]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]


** Dynamic Sequence Lengths **


```python
#TODO make this independent of dimensions
def get_sequence_length(sequence_ids, pad_word_id=0, axis=1):
    '''
    Returns the sequence length, droping out all the padded tokens if the sequence is padded
    
    :param sequence_ids: Tensor(shape=[batch_size, doc_length])
    :param pad_word_id: 0 is default
    :return: Array of Document lengths of size batch_size
    '''
    flag = tf.greater(sequence_ids, pad_word_id)
    used = tf.cast(flag, tf.int32)
    length = tf.reduce_sum(used, axis)
    length = tf.cast(length, tf.int32)
    return length
```


```python
# MAX_DOC_LENGTHS = 4
# rand_array = np.random.randint(1,MAX_DOC_LENGTHS, size=(3,5,4))

#Assume all negative values are padding
rand_array = np.array([[ 2,  0,  0,  0,  0,  0],
 [ 3,  4,  0,  0,  0,  0],
 [ 5,  6,  4,  0,  0,  0],
 [ 7,  8,  6,  4,  0,  0],
 [ 9, 10,  6, 11, 12, 13],
 [ 0,  0,  0,  0,  0,  0]])

rand_array

```




    array([[ 2,  0,  0,  0,  0,  0],
           [ 3,  4,  0,  0,  0,  0],
           [ 5,  6,  4,  0,  0,  0],
           [ 7,  8,  6,  4,  0,  0],
           [ 9, 10,  6, 11, 12, 13],
           [ 0,  0,  0,  0,  0,  0]])




```python
with tf.Session() as sess:
        length = get_sequence_length(rand_array, axis=1, pad_word_id=PAD_WORD_ID)
        print("Get dynamic sequence lengths: ", sess.run(length))
```

    Get dynamic sequence lengths:  [1 2 3 4 6 0]



```python
# data = np.random.randint(1,6, size=(3,MAX_DOCUMENT_LENGTH,4))

# print(data)
# with tf.Session() as sess:
#         length = get_sequence_length(data, axis=1)
#         print("Get dynamic sequence lengths: ", sess.run(length))
    
```


```python
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


def pad_sequences(sequences, pad_tok, nlevels, MAX_WORD_LENGTH=6):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        sequence_padded = []
        sequence_length = []
        max_length = max(map(lambda x : len(x.split(" ")), sequences))
        # sequence_padded, sequence_length = _pad_sequences(sequences,
        #                                                   pad_tok, max_length)
        #breaking the code to pad the string instead on its ids
        for seq in sequences:
            current_length = len(seq.split(" "))
            diff = max_length - current_length
            pad_data = pad_tok * diff
            sequence_padded.append(seq + pad_data)
            sequence_length.append(max_length) #assumed

        # print_info(sequence_length)
    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq))
        #                        for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in tqdm(sequences):
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, MAX_WORD_LENGTH)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*MAX_WORD_LENGTH,
                                            max_length_sentence) #TODO revert -1 to pad_tok
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length
```

## 1. Data Cleaning /Precrocessing


```python
# Assume each line to be an document
lines = ['Simple',
         'Some title', 
         'A longer title', 
         'An even longer title', 
         'This is longer than doc length isnt',
          '']
```

## 2. Extracting Vocab from the Corpus


```python
! rm vocab_test.tsv
```


```python
tf.reset_default_graph()


print ('TensorFlow Version: ', tf.__version__)


# Create vocabulary
# min_frequency -> consider a word if and only it repeats for fiven count
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, 
                                                                     min_frequency=0)
vocab_processor.fit(lines)

word_vocab = []

#Create a file and store the words
with gfile.Open('vocab_test.tsv', 'wb') as f:
    f.write("{}\n".format(PAD_WORD))
    word_vocab.append(PAD_WORD)
    for word, index in vocab_processor.vocabulary_._mapping.items():
        word_vocab.append(word)
        f.write("{}\n".format(word))
        
VOCAB_SIZE = len(vocab_processor.vocabulary_) + 1
print ('{} words into vocab.tsv'.format(VOCAB_SIZE))


id_2_word = {id:word for id, word in enumerate(word_vocab)}
id_2_word
```

    TensorFlow Version:  1.4.0
    15 words into vocab.tsv





    {0: '<PAD>',
     1: '<UNK>',
     2: 'Simple',
     3: 'Some',
     4: 'title',
     5: 'A',
     6: 'longer',
     7: 'An',
     8: 'even',
     9: 'This',
     10: 'is',
     11: 'than',
     12: 'doc',
     13: 'length',
     14: 'isnt'}




```python
! cat vocab_test.tsv
```

    <PAD>
    <UNK>
    Simple
    Some
    title
    A
    longer
    An
    even
    This
    is
    than
    doc
    length
    isnt



```python
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
    return final_vocab
```


```python
def get_char_vocab(words_vocab):
    '''

    :param words_vocab: List of words
    :return:
    '''
    words_chars_vocab = ['<P>', '<U>']
    chars = set()
    for word in words_vocab:
        for char in word:
            chars.add(str(char))
    words_chars_vocab.extend(chars)
    return sorted(chars)


chars = get_char_vocab(word_vocab)
# Create char2id map
char_2_id_map = {c:i for i,c in enumerate(chars)}

CHAR_VOCAB_SIZE = len(char_2_id_map)
char_2_id_map
```




    {'<': 0,
     '>': 1,
     'A': 2,
     'D': 3,
     'K': 4,
     'N': 5,
     'P': 6,
     'S': 7,
     'T': 8,
     'U': 9,
     'a': 10,
     'c': 11,
     'd': 12,
     'e': 13,
     'g': 14,
     'h': 15,
     'i': 16,
     'l': 17,
     'm': 18,
     'n': 19,
     'o': 20,
     'p': 21,
     'r': 22,
     's': 23,
     't': 24,
     'v': 25}



## 3. Preparing the features


```python
list_char_ids = []
char_ids_feature2 = []

for line in lines:
    for word in line.split():
        word_2_char_ids = [char_2_id_map.get(c, 0) for c in word]
        list_char_ids.append(word_2_char_ids)
    char_ids_feature2.append(list_char_ids)
    list_char_ids = []
```


```python
char_ids_feature2
```




    [[[7, 16, 18, 21, 17, 13]],
     [[7, 20, 18, 13], [24, 16, 24, 17, 13]],
     [[2], [17, 20, 19, 14, 13, 22], [24, 16, 24, 17, 13]],
     [[2, 19], [13, 25, 13, 19], [17, 20, 19, 14, 13, 22], [24, 16, 24, 17, 13]],
     [[8, 15, 16, 23],
      [16, 23],
      [17, 20, 19, 14, 13, 22],
      [24, 15, 10, 19],
      [12, 20, 11],
      [17, 13, 19, 14, 24, 15],
      [16, 23, 19, 24]],
     []]




```python
char_ids_feature2, char_seq_length = pad_sequences(char_ids_feature2, nlevels=2, pad_tok=0)
char_ids_feature2 = np.array(char_ids_feature2)
char_ids_feature2.shape
```

    100%|██████████| 6/6 [00:00<00:00, 34192.70it/s]





    (6, 7, 6)




```python
char_ids_feature2
```




    array([[[ 7, 16, 18, 21, 17, 13],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 7, 20, 18, 13,  0,  0],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 2,  0,  0,  0,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 2, 19,  0,  0,  0,  0],
            [13, 25, 13, 19,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 8, 15, 16, 23,  0,  0],
            [16, 23,  0,  0,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 15, 10, 19,  0,  0],
            [12, 20, 11,  0,  0,  0],
            [17, 13, 19, 14, 24, 15],
            [16, 23, 19, 24,  0,  0]],
    
           [[ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]]])




```python
print("Character IDs shape: ", char_ids_feature2.shape)
print("Number of sentences: ", len(lines))
print("MAX_DOC_LENGTH: ", max([len(line.split()) for line in lines]))
print("MAX_WORD_LENGTH: ", 8)
```

    Character IDs shape:  (6, 7, 6)
    Number of sentences:  6
    MAX_DOC_LENGTH:  7
    MAX_WORD_LENGTH:  8


## 4. Model


```python
tf.reset_default_graph()
```

#### Reference Link: https://www.tensorflow.org/api_docs/python/tf/contrib/lookup/index_table_from_file


```python
id_2_word
```




    {0: '<PAD>',
     1: '<UNK>',
     2: 'Simple',
     3: 'Some',
     4: 'title',
     5: 'A',
     6: 'longer',
     7: 'An',
     8: 'even',
     9: 'This',
     10: 'is',
     11: 'than',
     12: 'doc',
     13: 'length',
     14: 'isnt'}




```python
# can use the vocabulary to convert words to numbers
table = lookup.index_table_from_file(
  vocabulary_file='vocab_test.tsv', 
    num_oov_buckets=0, 
    vocab_size=None, 
    default_value=UNKNOWN_WORD_ID) #id of <PAD> is 0
```


```python
# string operations
# Array of Docs -> Split it into Tokens/words 
#               -> Convert it into Dense Tensor apending PADWORD
#               -> Table lookup 
#               -> Slice it to MAX_DOCUMENT_LENGTH
data = tf.constant(lines)
words = tf.string_split(data)

densewords = tf.sparse_tensor_to_dense(words, default_value=PAD_WORD)
word_ids = table.lookup(densewords)
print(word_ids)


##Following extrasteps are taken care by above 'table.lookup'
# now pad out with zeros and then slice to constant length
# padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
# # this takes care of documents with zero length also
# padded = tf.pad(word_ids, padding)

#if you wanted to clip the document MAX size then it can be done here!
# sliced = tf.slice(word_ids, [0,0], [-1, MAX_DOCUMENT_LENGTH])
```

    Tensor("hash_table_Lookup:0", shape=(?, ?), dtype=int64)



```python

word2ids = table.lookup(tf.constant(lines[1].split()))
word2ids_1 = table.lookup(tf.constant("Some unknown title".split()))


with tf.Session() as sess:
    #Tables needs to be initialized before using it
    tf.tables_initializer().run()
    print ("{} --> {}".format(lines[1], word2ids.eval()))
    print ("{} --> {}".format("Some unknown title", word2ids_1.eval()))
    print(sess.run(densewords))

```

    Some title --> [3 4]
    Some unknown title --> [3 1 4]
    [[b'Simple' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>']
     [b'Some' b'title' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>']
     [b'A' b'longer' b'title' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>']
     [b'An' b'even' b'longer' b'title' b'<PAD>' b'<PAD>' b'<PAD>']
     [b'This' b'is' b'longer' b'than' b'doc' b'length' b'isnt']
     [b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>' b'<PAD>']]



```python
seq_length= get_sequence_length(word_ids)
```


```python
with tf.Session() as sess:
     #Tables needs to be initialized before using it
    tf.tables_initializer().run()
    print(sess.run(word_ids))
    print("==========================")
    print(sess.run(seq_length))
```

    [[ 2  0  0  0  0  0  0]
     [ 3  4  0  0  0  0  0]
     [ 5  6  4  0  0  0  0]
     [ 7  8  6  4  0  0  0]
     [ 9 10  6 11 12 13 14]
     [ 0  0  0  0  0  0  0]]
    ==========================
    [1 2 3 4 7 0]


### Embed Layer


```python
with tf.device('/cpu:0'), tf.name_scope("embed-layer"):  

    # layer to take the words and convert them into vectors (embeddings)
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into
    # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
    word_embeddings = tf.contrib.layers.embed_sequence(word_ids,
                                              vocab_size=VOCAB_SIZE,
                                              embed_dim=WORD_EMBEDDING_SIZE,
                                                   initializer=tf.contrib.layers.xavier_initializer(
                                                                   seed=42))
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     #Tables needs to be initialized before using it
    tf.tables_initializer().run()
    print("word_embeddings : [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE] ", sess.run(word_embeddings).shape)
    print("<===============================================>")
    print(sess.run(word_embeddings))

```

    word_embeddings : [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE]  (6, 7, 3)
    <===============================================>
    [[[-0.36255243 -0.44535902 -0.18911475]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]]
    
     [[ 0.2578851  -0.3242403   0.41261792]
      [ 0.37403101  0.11017311 -0.57392919]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]]
    
     [[-0.29184508  0.00701374 -0.15982357]
      [-0.52557528  0.54521036  0.37919033]
      [ 0.37403101  0.11017311 -0.57392919]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]]
    
     [[-0.09862986  0.11739373 -0.18522915]
      [ 0.08441174 -0.30454588  0.20182377]
      [-0.52557528  0.54521036  0.37919033]
      [ 0.37403101  0.11017311 -0.57392919]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]]
    
     [[-0.56889766 -0.55320239  0.53333259]
      [ 0.15531081 -0.31604788  0.17497867]
      [-0.52557528  0.54521036  0.37919033]
      [ 0.4238075  -0.28343284 -0.2626003 ]
      [-0.04067117 -0.24000475 -0.33476758]
      [-0.08533373  0.50026119 -0.095366  ]
      [ 0.24810773  0.07649624 -0.23006374]]
    
     [[ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]
      [ 0.52223814  0.20485282  0.34100413]]]



```python
with  tf.name_scope("word_level_lstm_layer"):
    # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
    d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                state_is_tuple=True)
    #https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
    d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(WORD_LEVEL_LSTM_HIDDEN_SIZE,
                                                state_is_tuple=True)

    (fw_output_one, bw_output_one), output_states = tf.nn.bidirectional_dynamic_rnn(
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
```

    INFO:tensorflow:encoded_sentence =====> Tensor("word_level_lstm_layer/concat:0", shape=(?, ?, 6), dtype=float32)



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     #Tables needs to be initialized before using it
    tf.tables_initializer().run()
    
    tf.logging.info('fw_output_one \t=====> {}'.format(fw_output_one.get_shape()))
    tf.logging.info('bw_output_one \t=====> {}'.format(bw_output_one.get_shape()))
    tf.logging.info("=================================================================")
    tf.logging.info('forward hidden state \t=====> {}'.format(output_states[0][0].get_shape()))
    tf.logging.info('forward out state \t=====> {}'.format(output_states[0][1].get_shape()))
    tf.logging.info('backward hidden state \t=====> {}'.format(output_states[1][0].get_shape()))
    tf.logging.info('backward out state \t=====> {}'.format(output_states[1][1].get_shape()))
    tf.logging.info("=================================================================")
    tf.logging.info('encoded_sentence \t=====> {}'.format(encoded_sentence.get_shape()))
    tf.logging.info("=================================================================")
    encoded_senence_out =  encoded_sentence.eval()
    #check for zeros in the encoced sentence, where it omits padded words
    tf.logging.info('encoded_senence_out \t=====> {}'.format(encoded_senence_out.shape))
    tf.logging.info("=================================================================")
    print("encoded_senence_out:\n" , encoded_senence_out)
    
```

    INFO:tensorflow:fw_output_one 	=====> (?, ?, 3)
    INFO:tensorflow:bw_output_one 	=====> (?, ?, 3)
    INFO:tensorflow:=================================================================
    INFO:tensorflow:forward hidden state 	=====> (?, 3)
    INFO:tensorflow:forward out state 	=====> (?, 3)
    INFO:tensorflow:backward hidden state 	=====> (?, 3)
    INFO:tensorflow:backward out state 	=====> (?, 3)
    INFO:tensorflow:=================================================================
    INFO:tensorflow:encoded_sentence 	=====> (?, ?, 6)
    INFO:tensorflow:=================================================================
    INFO:tensorflow:encoded_senence_out 	=====> (6, 7, 6)
    INFO:tensorflow:=================================================================
    encoded_senence_out:
     [[[-0.01976291 -0.00923283  0.07085376  0.01306459  0.03344805 -0.04371169]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[-0.04772531  0.11881609 -0.03189922 -0.03764752 -0.07253012  0.04396541]
      [ 0.03150441  0.07321765  0.00564955 -0.08588977  0.0064748   0.05586476]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.00638387 -0.04680856  0.03509862 -0.00451156  0.07471974 -0.05366353]
      [-0.01397579 -0.11594137 -0.04197412 -0.00452052  0.06904873 -0.04156622]
      [ 0.07882241 -0.11152568 -0.01332429 -0.08588977  0.0064748   0.05586476]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.02174438 -0.0365928   0.01485884 -0.00045621  0.02587043 -0.01915273]
      [-0.01144159  0.03617312  0.01070661  0.01835279 -0.00727936 -0.01345707]
      [-0.03788993 -0.0718103  -0.05404014 -0.00452052  0.06904873 -0.04156622]
      [ 0.05784972 -0.06690629 -0.01731192 -0.08588977  0.0064748   0.05586476]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[-0.10292543  0.03962895  0.03830826  0.10795461 -0.02236171 -0.08061882]
      [-0.11615913  0.11212726  0.0149015  -0.00817155 -0.02850961 -0.01442205]
      [-0.12170991 -0.030709   -0.05609156 -0.02677963  0.04574642 -0.05044449]
      [-0.05674711  0.04917112 -0.01982559 -0.1282053  -0.01708856  0.04558847]
      [-0.02428005  0.03635133  0.03372234 -0.07728597  0.03857267  0.00894586]
      [ 0.00910437 -0.04255038 -0.00274513 -0.05090441  0.03855459  0.02097071]
      [ 0.04807584 -0.03034803 -0.00613315 -0.04632962 -0.00689608  0.03440535]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]]



```python
print(char_ids_feature2.shape)
char_ids_feature2
```

    (6, 7, 6)





    array([[[ 7, 16, 18, 21, 17, 13],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 7, 20, 18, 13,  0,  0],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 2,  0,  0,  0,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 2, 19,  0,  0,  0,  0],
            [13, 25, 13, 19,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 16, 24, 17, 13,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]],
    
           [[ 8, 15, 16, 23,  0,  0],
            [16, 23,  0,  0,  0,  0],
            [17, 20, 19, 14, 13, 22],
            [24, 15, 10, 19,  0,  0],
            [12, 20, 11,  0,  0,  0],
            [17, 13, 19, 14, 24, 15],
            [16, 23, 19, 24,  0,  0]],
    
           [[ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0]]])




```python
with tf.variable_scope("char_embed_layer"):
    
        char_ids = tf.convert_to_tensor(char_ids_feature2, np.int64)
        s = tf.shape(char_ids)
        #remove pad words
        char_ids_reshaped = tf.reshape(char_ids, shape=(s[0] * s[1], s[2])) #6 -> char dim
        
        char_embeddings = tf.contrib.layers.embed_sequence(char_ids,
                                                           vocab_size=CHAR_VOCAB_SIZE,
                                                           embed_dim=CHAR_EMBEDDING_SIZE,
                                                           initializer=tf.contrib.layers.xavier_initializer(
                                                               seed=42))
        word_lengths = get_sequence_length(char_ids_reshaped)

        #[BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
        tf.logging.info('char_ids_reshaped =====> {}'.format(char_ids_reshaped))
        tf.logging.info('char_embeddings =====> {}'.format(char_embeddings))
```

    INFO:tensorflow:char_ids_reshaped =====> Tensor("char_embed_layer/Reshape:0", shape=(?, ?), dtype=int64)
    INFO:tensorflow:char_embeddings =====> Tensor("char_embed_layer/EmbedSequence/embedding_lookup:0", shape=(6, 7, 6, 3), dtype=float32)



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
     #Tables needs to be initialized before using it
    tf.tables_initializer().run()
    tf.logging.info('char_ids =====> {}'.format(char_ids_reshaped.get_shape()))
    tf.logging.info("=================================================================")
    res = char_ids_reshaped.eval()
    tf.logging.info('char_ids_reshaped shape =====> {}\n'.format(res.shape))
    tf.logging.info('char_ids_reshaped =====> {}\n'.format(res))
    tf.logging.info('word_lengths =====> {}\n'.format(word_lengths.eval()))
    tf.logging.info("=================================================================")
    tf.logging.info('char_embeddings =====> {}'.format(char_embeddings.shape))
    tf.logging.info("=================================================================")
    char_embeddings_out = char_embeddings.eval()
    print(char_embeddings_out.shape)
    tf.logging.info("=================================================================")
    char_embeddings_out = char_embeddings.eval()
    print("char_embeddings_out shape\n", char_embeddings_out.shape)
    print("char_embeddings_out \n", char_embeddings_out)
```

    INFO:tensorflow:char_ids =====> (?, ?)
    INFO:tensorflow:=================================================================
    INFO:tensorflow:char_ids_reshaped shape =====> (42, 6)
    
    INFO:tensorflow:char_ids_reshaped =====> [[ 7 16 18 21 17 13]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 7 20 18 13  0  0]
     [24 16 24 17 13  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 2  0  0  0  0  0]
     [17 20 19 14 13 22]
     [24 16 24 17 13  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 2 19  0  0  0  0]
     [13 25 13 19  0  0]
     [17 20 19 14 13 22]
     [24 16 24 17 13  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 8 15 16 23  0  0]
     [16 23  0  0  0  0]
     [17 20 19 14 13 22]
     [24 15 10 19  0  0]
     [12 20 11  0  0  0]
     [17 13 19 14 24 15]
     [16 23 19 24  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]
     [ 0  0  0  0  0  0]]
    
    INFO:tensorflow:word_lengths =====> [6 0 0 0 0 0 0 4 5 0 0 0 0 0 1 6 5 0 0 0 0 2 4 6 5 0 0 0 4 2 6 4 3 6 4 0 0
     0 0 0 0 0]
    
    INFO:tensorflow:=================================================================
    INFO:tensorflow:char_embeddings =====> (6, 7, 6, 3)
    INFO:tensorflow:=================================================================
    (6, 7, 6, 3)
    INFO:tensorflow:=================================================================
    char_embeddings_out shape
     (6, 7, 6, 3)
    char_embeddings_out 
     [[[[-0.0777044   0.09248734 -0.14593068]
       [-0.37224245 -0.19546646  0.22445828]
       [ 0.06220943 -0.08555388  0.28730118]
       [ 0.09594625  0.33095586 -0.30559146]
       [ 0.33953965 -0.07726747  0.38769937]
       [-0.06722921  0.39412504 -0.07513303]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]
    
    
     [[[-0.0777044   0.09248734 -0.14593068]
       [-0.06029078  0.31312287  0.20676911]
       [ 0.06220943 -0.08555388  0.28730118]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[-0.21950163 -0.27822471  0.44486523]
       [-0.37224245 -0.19546646  0.22445828]
       [-0.21950163 -0.27822471  0.44486523]
       [ 0.33953965 -0.07726747  0.38769937]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]
    
    
     [[[-0.28563282 -0.35087106 -0.14899191]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.33953965 -0.07726747  0.38769937]
       [-0.06029078  0.31312287  0.20676911]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.19546884  0.06026673 -0.18125311]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.14145756  0.13300532  0.06176662]]
    
      [[-0.21950163 -0.27822471  0.44486523]
       [-0.37224245 -0.19546646  0.22445828]
       [-0.21950163 -0.27822471  0.44486523]
       [ 0.33953965 -0.07726747  0.38769937]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]
    
    
     [[[-0.28563282 -0.35087106 -0.14899191]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[-0.06722921  0.39412504 -0.07513303]
       [-0.34800512 -0.36377969  0.4280228 ]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.33953965 -0.07726747  0.38769937]
       [-0.06029078  0.31312287  0.20676911]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.19546884  0.06026673 -0.18125311]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.14145756  0.13300532  0.06176662]]
    
      [[-0.21950163 -0.27822471  0.44486523]
       [-0.37224245 -0.19546646  0.22445828]
       [-0.21950163 -0.27822471  0.44486523]
       [ 0.33953965 -0.07726747  0.38769937]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]
    
    
     [[[ 0.06650281 -0.23993301  0.15900457]
       [ 0.31817758 -0.04265583 -0.45091799]
       [-0.37224245 -0.19546646  0.22445828]
       [-0.32428217 -0.32085785  0.26928455]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[-0.37224245 -0.19546646  0.22445828]
       [-0.32428217 -0.32085785  0.26928455]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.33953965 -0.07726747  0.38769937]
       [-0.06029078  0.31312287  0.20676911]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.19546884  0.06026673 -0.18125311]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.14145756  0.13300532  0.06176662]]
    
      [[-0.21950163 -0.27822471  0.44486523]
       [ 0.31817758 -0.04265583 -0.45091799]
       [ 0.12235987 -0.24899472  0.13785499]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[-0.03204235 -0.18908501 -0.26374283]
       [-0.06029078  0.31312287  0.20676911]
       [ 0.33389199 -0.22329932 -0.20688666]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.33953965 -0.07726747  0.38769937]
       [-0.06722921  0.39412504 -0.07513303]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [ 0.19546884  0.06026673 -0.18125311]
       [-0.21950163 -0.27822471  0.44486523]
       [ 0.31817758 -0.04265583 -0.45091799]]
    
      [[-0.37224245 -0.19546646  0.22445828]
       [-0.32428217 -0.32085785  0.26928455]
       [ 0.38928831 -0.23052102 -0.0280644 ]
       [-0.21950163 -0.27822471  0.44486523]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]
    
    
     [[[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]
    
      [[ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]
       [ 0.41143936  0.16139096  0.26865625]]]]



```python
with tf.variable_scope("chars_level_bilstm_layer"):
        # put the time dimension on axis=1
        shape = tf.shape(char_embeddings)

        BATCH_SIZE = shape[0]
        MAX_DOC_LENGTH = shape[1]
        CHAR_MAX_LENGTH = shape[2]

        # [BATCH_SIZE, MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]  ===>
        #      [BATCH_SIZE * MAX_SEQ_LENGTH, MAX_WORD_LEGTH, CHAR_EMBEDDING_SIZE]
        char_embeddings_reshaped = tf.reshape(char_embeddings,
                                     shape=[BATCH_SIZE * MAX_DOC_LENGTH, CHAR_MAX_LENGTH,
                                            CHAR_EMBEDDING_SIZE],
                                     name="reduce_dimension_1")

        tf.logging.info('reshaped char_embeddings =====> {}'.format(char_embeddings))

        # word_lengths = get_sequence_length_old(char_embeddings) TODO working
        word_lengths = get_sequence_length(char_ids_reshaped)

        tf.logging.info('word_lengths =====> {}'.format(word_lengths))

        # bi lstm on chars
        cell_fw = tf.contrib.rnn.LSTMCell(CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                          state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(CHAR_LEVEL_LSTM_HIDDEN_SIZE,
                                          state_is_tuple=True)

        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            dtype=tf.float32,
            sequence_length=word_lengths,
            inputs=char_embeddings_reshaped,
            scope="encode_words")

        # read and concat output
        (char_fw_output_one, char_bw_output_one) , output_state = _output
        ((hidden_fw, output_fw), (hidden_bw, output_bw)) = output_state
        encoded_words = tf.concat([output_fw, output_bw], axis=-1)
        
        char_encoded = tf.concat([char_fw_output_one,
                                  char_bw_output_one], axis=-1)
        lstm_out_encoded_words = encoded_words
        # [BATCH_SIZE, MAX_SEQ_LENGTH, WORD_EMBEDDING_SIZE]
        encoded_words = tf.reshape(encoded_words,
                                   shape=[BATCH_SIZE, MAX_DOC_LENGTH, 2 *
                                          CHAR_LEVEL_LSTM_HIDDEN_SIZE])

        tf.logging.info('encoded_words =====> {}'.format(encoded_words))


    
    
    
```

    INFO:tensorflow:reshaped char_embeddings =====> Tensor("char_embed_layer/EmbedSequence/embedding_lookup:0", shape=(6, 7, 6, 3), dtype=float32)
    INFO:tensorflow:word_lengths =====> Tensor("chars_level_bilstm_layer/Sum:0", shape=(?,), dtype=int32)
    INFO:tensorflow:encoded_words =====> Tensor("chars_level_bilstm_layer/Reshape:0", shape=(?, ?, 6), dtype=float32)



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.tables_initializer().run()
    
    tf.logging.info('char_embeddings =====> {}'.format(char_embeddings.shape))
    
    tf.logging.info('char_encoded =====> {}'.format(char_encoded.get_shape()))
    print("char_encoded:\n", char_encoded.eval())
    
    tf.logging.info('char_fw_output_one =====> {}'.format(char_fw_output_one.get_shape()))
    tf.logging.info('char_bw_output_one =====> {}'.format(char_bw_output_one.get_shape()))
    
    tf.logging.info('char hidden_fw =====> {}'.format(hidden_fw.get_shape()))
    tf.logging.info('char output_fw =====> {}'.format(output_fw.get_shape()))
    tf.logging.info('char hidden_bw =====> {}'.format(hidden_bw.get_shape()))
    tf.logging.info('char output_bw =====> {}'.format(output_bw.get_shape()))
    
    tf.logging.info('lstm_out_encoded_words =====> {}'.format(lstm_out_encoded_words.get_shape()))
    #check for zeros in the encoced words, where it omits padded characters
    tf.logging.info('lstm_out_encoded_words =====> {}\n'.format(lstm_out_encoded_words.eval()))
    tf.logging.info('=====================================================================')
    tf.logging.info('encoded_words =====> {}'.format(encoded_words.get_shape()))
    tf.logging.info('encoded_words =====> {}\n'.format(encoded_words.eval()))
    
```

    INFO:tensorflow:char_embeddings =====> (6, 7, 6, 3)
    INFO:tensorflow:char_encoded =====> (?, ?, 6)
    char_encoded:
     [[[ 0.01289693  0.03215854  0.0049089   0.02896848  0.01639003  0.01760693]
      [-0.00175623 -0.0293929   0.0057649  -0.00010866 -0.02562934  0.02795034]
      [-0.02921278 -0.06415376 -0.00687395 -0.00772599 -0.02570577 -0.00648446]
      [-0.00780411  0.04103209 -0.00176386  0.03748152  0.03362147 -0.01367662]
      [-0.05726055 -0.03375445 -0.02974403 -0.02291263 -0.04749316 -0.00541486]
      [-0.06904165  0.03736177 -0.03910471  0.06656165  0.05170256  0.00199127]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     ..., 
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]]
    INFO:tensorflow:char_fw_output_one =====> (?, ?, 3)
    INFO:tensorflow:char_bw_output_one =====> (?, ?, 3)
    INFO:tensorflow:char hidden_fw =====> (?, 3)
    INFO:tensorflow:char output_fw =====> (?, 3)
    INFO:tensorflow:char hidden_bw =====> (?, 3)
    INFO:tensorflow:char output_bw =====> (?, 3)
    INFO:tensorflow:lstm_out_encoded_words =====> (?, 6)
    INFO:tensorflow:lstm_out_encoded_words =====> [[-0.06904165  0.03736177 -0.03910471  0.02896848  0.01639003  0.01760693]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [-0.08010477  0.04053386 -0.04397502  0.06615115  0.02842443  0.00285061]
     [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.0522683  -0.0173936   0.0234098   0.00279257  0.01505784  0.00404977]
     [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
     [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.06193589 -0.03623288  0.02355442 -0.04486501 -0.01258224  0.00184671]
     [-0.0254913  -0.01338712 -0.01496682  0.04904424  0.03080448  0.02878405]
     [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
     [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.02549884 -0.0902979   0.0286562  -0.06195    -0.05064696  0.01262299]
     [-0.00716779 -0.09858887  0.01202889 -0.01120201 -0.03311117  0.05316196]
     [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
     [ 0.04839797 -0.04648409  0.01917223 -0.10114393 -0.06143828  0.0299458 ]
     [ 0.03395079  0.0052877   0.00589937 -0.00152577  0.01732808 -0.00922754]
     [ 0.03923708  0.01564591  0.00568556 -0.08271034 -0.05915968  0.00373549]
     [-0.00868067 -0.13859105  0.01685495 -0.07438852 -0.06129534  0.08159567]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.        ]]
    
    INFO:tensorflow:=====================================================================
    INFO:tensorflow:encoded_words =====> (?, ?, 6)
    INFO:tensorflow:encoded_words =====> [[[-0.06904165  0.03736177 -0.03910471  0.02896848  0.01639003  0.01760693]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[-0.08010477  0.04053386 -0.04397502  0.06615115  0.02842443  0.00285061]
      [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.0522683  -0.0173936   0.0234098   0.00279257  0.01505784  0.00404977]
      [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
      [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.06193589 -0.03623288  0.02355442 -0.04486501 -0.01258224  0.00184671]
      [-0.0254913  -0.01338712 -0.01496682  0.04904424  0.03080448  0.02878405]
      [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
      [-0.06824481 -0.03029683 -0.01961379 -0.07010335 -0.10903811  0.08128914]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]
    
     [[ 0.02549884 -0.0902979   0.0286562  -0.06195    -0.05064696  0.01262299]
      [-0.00716779 -0.09858887  0.01202889 -0.01120201 -0.03311117  0.05316196]
      [-0.05949131  0.03973037 -0.03993161 -0.05799981 -0.08668529 -0.00825493]
      [ 0.04839797 -0.04648409  0.01917223 -0.10114393 -0.06143828  0.0299458 ]
      [ 0.03395079  0.0052877   0.00589937 -0.00152577  0.01732808 -0.00922754]
      [ 0.03923708  0.01564591  0.00568556 -0.08271034 -0.05915968  0.00373549]
      [-0.00868067 -0.13859105  0.01685495 -0.07438852 -0.06129534  0.08159567]]
    
     [[ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]
      [ 0.          0.          0.          0.          0.          0.        ]]]
    



```python

```

# References: 
- https://medium.com/towards-data-science/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575
- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/textclassification


```python

```
