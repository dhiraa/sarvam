import sys
import numpy as np
import  argparse
sys.path.append("../")

from word_vec.utils.downloader import *
from word_vec.utils.common import vocab_to_tsv
from word_vec.word2vec import Word2Vec, Word2VecConfig
from word_vec.utils.glove_data_iterator import setup_input_graph

def get_model(opt):
    config = Word2VecConfig(vocab_size=opt.vocab_size,
                            words_vocab_file="tmp/vocab.tsv",
                            embedding_size=opt.embed_size,
                            num_word_sample=64,
                            learning_rate=opt.learning_rate,
                            model_dir="tmp/model/")
    model = Word2Vec(config)
    return model

def word_vec(opt):
    text_path = download(FILE_NAME, EXPECTED_BYTES, "tmp/")
    words = read_data(file_path=text_path)
    word_2_id, id_2_word = build_vocab(words=words, vocab_size=50000)
    top_n_words = word_2_id.keys()
    vocab_to_tsv(vocab_list=top_n_words, outfilename="tmp/vocab.tsv")

    features, labels = generate_sample(list(top_n_words),
                                       context_window_size=3)

    features = np.asarray(features)
    labels = np.asarray(labels)

    model = get_model(opt)

    input_fn, intput_hook = setup_input_graph(features, labels, 128)

    NUM_EPOCHS = int(opt.num_epochs)
    BATCH_SIZE = int(opt.batch_size)
    MAX_STEPS = (features.shape[0] // BATCH_SIZE) * NUM_EPOCHS

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

    optparse.add_argument('-es', '--embed_size', action='store',
                          dest='embed_size', required=False,
                          default=64,
                          help='Word2Vec embedding size')

    optparse.add_argument('-lr', '--learning_rate', action='store',
                          dest='learning_rate', required=False,
                          default=0.001,
                          help='Learning Rate (0.001)')

    word_vec(opt = optparse.parse_args())