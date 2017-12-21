import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from nltk.corpus import stopwords

def num_words(text):
    text_splited = str(text).split(' ')
    return len(text_splited)

def num_upper_words(text):
    num = len([w for w in str(text).split() if w.isupper()])
    return num

def num_title_words(text):
    num = len([w for w in str(text).split() if w.istitle()])
    return num

def mean_word_len(text):
    return np.mean([len(w) for w in str(text).split()])

def unique_word_fraction(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    return (unique_count/word_count)


eng_stopwords = set(stopwords.words("english"))
def stopwords_count(text):
    """ Number of stopwords fraction in a text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    stopwords_count = len([w for w in text_splited if w in eng_stopwords])
    return (stopwords_count/word_count)


def punctuations_fraction(text):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    num_chars = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return (punctuation_count/num_chars)


def num_chars(text):
    """function to return number of characters """
    return len(text)

def fraction_noun(text):
    """function to give us fraction of noun over total words """
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    return (noun_count/word_count)

def fraction_adj(text):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

def fraction_verbs(text):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/word_count)



def append_meta_features(df, text_col):
    '''
    Adds following meta features to the dataframe based on given text column
     - num_words
     - max_word_length
     - num_upper_words
     - num_title_words
     - num_chars
     - mean_word_len
     - unique_word_fraction
     - stopwords_fraction
     - punctuations_fraction
     - fraction_noun
     - fraction_adj
     - fraction_verbs
     
    :param df: 
    :param text_col: 
    :return: 
    '''
    df["num_words"] = df[text_col].apply(num_words)
    df["max_word_length"] = df[text_col].apply(lambda sentence:
                                                 max([len(word) \
                                                      for word in sentence.split(" ")]))
    df["num_upper_words"] = df[text_col].apply(num_upper_words)
    df["num_title_words"] = df[text_col].apply(num_title_words)
    df['num_chars'] = df[text_col].apply(lambda row: num_chars(row))
    df["mean_word_len"] = df[text_col].apply(mean_word_len)
    df['unique_word_fraction'] = df[text_col].apply(lambda row: unique_word_fraction(row))
    df['stopwords_fraction'] = df[text_col].apply(lambda row: stopwords_count(row))
    df['punctuations_fraction'] = df[text_col].apply(lambda row: punctuations_fraction(row))
    df['fraction_noun'] = df[text_col].apply(lambda row: fraction_noun(row))
    df['fraction_adj'] = df[text_col].apply(lambda row: fraction_adj(row))
    df['fraction_verbs'] = df[text_col].apply(lambda row: fraction_verbs(row))
    return df





