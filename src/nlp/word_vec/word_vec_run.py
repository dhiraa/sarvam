import sys
import numpy as np
import  argparse
sys.path.append("../")
import pickle
from word_vec.utils.downloader import *
from word_vec.utils.common import vocab_to_tsv
from word_vec.glove import *
from word_vec.skip_gram import *
from word_vec.utils.dataset import *
from word_vec.utils.data_iterator import skip_gram_iterator

TEXT_DIR = "tmp/text/"
SKIP_GRAM_TRAIN_FILE = "tmp/skip_gram_dataset.pickle"

def get_model(opt):
    if opt.model == "skip_gram":
        config = SkipGramConfig(vocab_size=int(opt.vocab_size),
                                words_vocab_file="tmp/vocab.tsv",
                                embedding_size=opt.embed_size,
                                num_word_sample=64,  #Negative sampling
                                learning_rate=int(opt.learning_rate),
                                model_dir="tmp/model/")
        model = SkipGram(config)
    else:
        model= ""
    return model

def word_vec(opt):
    if not os.path.exists(TEXT_DIR):
        print("!!! RUN setup.sh !!!")
        exit(0)

    if not os.path.exists(SKIP_GRAM_TRAIN_FILE):
        dataset = TextDataset(vocabulary_size=int(opt.vocab_size),
                               min_occurrences=5,
                               window_size=int(opt.window_size),
                               name='GloveDataset',
                               text_dir=TEXT_DIR)

        dataset.prepare()
        del dataset

    input("Press ENTER to continue to run the model...")
    print("Loading train data...")
    TRAIN_DATA = pickle.load(open(SKIP_GRAM_TRAIN_FILE, "rb"))

    model = get_model(opt)

    NUM_EXAMPLES = len(TRAIN_DATA["center_words"])
    NUM_EPOCHS = int(opt.num_epochs)
    BATCH_SIZE = int(opt.batch_size)
    MAX_STEPS = (NUM_EXAMPLES // BATCH_SIZE) * NUM_EPOCHS

    input_fn, intput_hook = skip_gram_iterator(TRAIN_DATA, BATCH_SIZE)

    store_hook = model.get_store_hook()

    model.train(input_fn=input_fn, hooks=[intput_hook, store_hook], steps=MAX_STEPS)

if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Word2Vec")

    optparse.add_argument('-m', '--model', action='store',
                          dest='model', required=False,
                          default="skip_gram",
                          help='skip_gram or glove')

    optparse.add_argument('-ne', '--num_epochs', action='store',
                          dest='num_epochs', required=False,
                          default=50,
                          help='Number of epochs for training')

    optparse.add_argument('-bs', '--batch_size', action='store',
                          dest='batch_size', required=False,
                          default=128,
                          help='Batch size for training/testing')

    optparse.add_argument('-vs', '--vocab_size', action='store',
                          dest='vocab_size', required=False,
                          default=50000,
                          help='Number of top vocab to be considered')

    optparse.add_argument('-ws', '--window_size', action='store',
                          dest='window_size', required=False,
                          default=1,
                          help='Window size for skip gram')


    optparse.add_argument('-es', '--embed_size', action='store',
                          dest='embed_size', required=False,
                          default=64,
                          help='Word2Vec embedding size')

    optparse.add_argument('-lr', '--learning_rate', action='store',
                          dest='learning_rate', required=False,
                          default=1.0,
                          help='Learning Rate (1.0)')

    word_vec(opt = optparse.parse_args())