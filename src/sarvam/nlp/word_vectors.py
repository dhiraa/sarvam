from tqdm import tqdm
import numpy as np
from sarvam.colorful_logger import *
##  Data : http://www-nlp.stanford.edu/data/glove.840B.300d.zip

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# load the GloVe vectors in a dictionary:
def load_embeddings(glove_text_path):
    embeddings_index = {}
    f = open(glove_text_path)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index

 #this function creates a normalized vector for the whole sentence
def sent2vec(s, embeddings_index):
    words = str(s)#.lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

def data_2_glove_vec(data, embeddings_index):
    '''
    
    :param data: List of text 
    :return: 
    '''
    # create sentence vectors using the above function for training and validation set
    sentence_vectors = [sent2vec(x, embeddings_index) for x in tqdm(data)]
    return np.array(sentence_vectors)

def spacy_word2vec(word, spacy_nlp):
    lex = spacy_nlp(word)
    if lex.has_vector:
        return lex.vector
    else:
        return spacy_nlp.vocab[0].vector  # return all zeros for Out of vocab

