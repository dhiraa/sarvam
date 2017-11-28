import sys
import numpy as np
import  argparse
sys.path.append("../")
import pickle
from word_vec.utils.downloader import *
from word_vec.utils.common import vocab_to_tsv
from word_vec.word2vec_v1 import Word2VecV1, Word2VecConfigV1
from word_vec.utils.glove_data_iterator import setup_input_graph
from word_vec.utils.preprocessing import *
def get_model(opt):
    config = Word2VecConfigV1(vocab_size=int(opt.vocab_size),
                            words_vocab_file="tmp/vocab.tsv",
                            embedding_size=opt.embed_size,
                            num_word_sample=64, #Negative sampling
                            learning_rate=opt.learning_rate,
                            model_dir="tmp/model/")
    model = Word2VecV1(config)
    return model

def word_vec(opt):

    TEXT_DIR = "tmp/text/"
    TRAIN_FILE = "tmp/train_data.pickle"

    if not os.path.exists(TEXT_DIR):
        print("!!! RUN setup.sh !!!")
        exit(0)

    if not os.path.exists(TRAIN_FILE):
        dataset = GloveDataset(vocabulary_size=int(opt.vocab_size),
                               min_occurrences=5,
                               window_size=int(opt.window_size),
                               name='GloveDataset',
                               text_dir=TEXT_DIR)

        dataset.prepare()
    else:
        TRAIN_DATA = pickle.load(open(TRAIN_FILE, "rb"))

    model = get_model(opt)

    NUM_EXAMPLES = TRAIN_DATA["words"].shape[0]
    NUM_EPOCHS = int(opt.num_epochs)
    BATCH_SIZE = int(opt.batch_size)
    MAX_STEPS = (NUM_EXAMPLES // BATCH_SIZE) * NUM_EPOCHS

    input_fn, intput_hook = setup_input_graph(TRAIN_DATA, BATCH_SIZE)
    store_hook = model.get_store_hook()

    model.train(input_fn=input_fn, hooks=[intput_hook, store_hook], steps=MAX_STEPS)

if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Prepare data for Tensorflow training...")

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
                          help='Learning Rate (0.001)')

    word_vec(opt = optparse.parse_args())