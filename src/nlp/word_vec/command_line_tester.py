
import numpy as np
from scipy.spatial import distance
import os
import sys
sys.path.append("../")

import word_vec.utils.common as common

EMBED_MAT_PATH = "tmp/embed_mat.npy"
VOCAB_PATH = "tmp/vocab.tsv"

embed_mat = np.load(EMBED_MAT_PATH)
d = np.sum(embed_mat ** 2, 1) ** 0.5
W_norm = (embed_mat.T / d).T

word_2_id, id_2_word = common.tsv_to_vocab(VOCAB_PATH)

def similarity(word1, word2):
    ind = word_2_id[word1]
    ind1 = word_2_id[word2]
    wordvec = W_norm[ind, :]
    wordvec1 = W_norm[ind1, :]
    sim = 1 - distance.cosine(wordvec, wordvec1)
    print(sim)


def evaluate(word):
    try:
        ind = word_2_id[word]
        wordvec = W_norm[ind, :]
        dist = [distance.cosine(wordvec, vec) for vec in W_norm]
        indices = np.argsort(dist)
        closest_words = [id_2_word[ind] for ind in indices]
        closest_words = closest_words[1:10]
        print(closest_words)
    except:
        print("Sorry!!! the word: {} you are looking for is not included in the vocab.".format(word))

if __name__ == "__main__":

    if not os.path.exists(EMBED_MAT_PATH) or not os.path.exists(VOCAB_PATH):
        print("Train the model, before you can play with it!!!")
        exit(0)

    print("Enter words and system will display similar words, enter 'exit' to exit")
    while True:
        inp = input()
        evaluate(inp)
        if inp == "exit":
        	break
