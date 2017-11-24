import nltk
from tqdm import tqdm_notebook as tqdm
import pickle
import  os
import spacy
import nltk
import pandas as pd
from tqdm import tqdm_notebook as tqdm
# Find how often each Category used each word


def tokenize(df: pd.DataFrame, text_col, nlp=spacy.load('en_core_web_md')):
    def cleaning(sentence):
        sentence = nlp(sentence)
        tokens = [token.text for token in sentence]
        tokens = ' '.join(tokens)
        return tokens

    df = df.assign(spacy_processed = lambda rows : rows[text_col].map(lambda row: cleaning(row)))

    return df

def extract_lemmas(df: pd.DataFrame, text_col, nlp=spacy.load('en')):
    stopwords = nltk.corpus.stopwords.words('english')

    def cleaning(sentence):
        sentence = nlp(sentence)
        tokens = [token.lemma_+"_"+token.pos_ for token in sentence
                  if not token.is_punct | token.is_space | token.is_bracket | (token.text in stopwords)]
        tokens = ' '.join(tokens)
        return tokens

    df = df.assign(nlp_processed = lambda rows : rows[text_col].map(lambda row: cleaning(row)))

    return df

def doc_to_integers(df: pd.DataFrame, text_col, nlp=spacy.load('en')):
    stopwords = nltk.corpus.stopwords.words('english')

    def cleaning(sentence):
        sentence = nlp(sentence)
        tokens = [token.orth + token.lemma + token.pos for token in sentence if not token.is_punct | token.is_space | token.is_bracket | (token.text in stopwords)]
        return tokens

    df = df.assign(nlp_processed = lambda rows : rows[text_col].map(lambda row: cleaning(row)))

    return df

#Refence: https://www.kaggle.com/mageswaran/beginner-s-tutorial-python/editnb
class ConditionalWordFreq:

    def __init__(self):
        # word frequency by category
        self.wordFreqByCategory = None


    def fit(self, df, tex_col, category_col):

        self.wordFreqByCategory = nltk.probability.ConditionalFreqDist()

        by_category = df.groupby(category_col)
        for category, group in tqdm(by_category):
            sentences = group[tex_col].str.cat(sep = ' ')

            sentences = sentences.lower()

            tokens = nltk.tokenize.word_tokenize(sentences)

            frequency = nltk.FreqDist(tokens)

            self.wordFreqByCategory[category] = (frequency)

        wordFreqByCategoryFile = open('wordFreqByCategory.pickle', 'wb')
        pickle.dump(self.wordFreqByCategory, wordFreqByCategoryFile)

    def load(self):
        if os.path.exists('wordFreqByCategory.pickle') and self.wordFreqByCategory is None:
            self.wordFreqByCategory = pickle.load(open('wordFreqByCategory.pickle', 'rb'))

    def category_probabilities(self, word):
        for category in self.wordFreqByCategory.keys():
            print('Probability of' + word + '  in ' + category + ' is ' + str(self.wordFreqByCategory[category].freq(word)))

