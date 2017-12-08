from word_vec.utils.dataset import *

TEXT_DIR = "tmp/text/"
TRAIN_FILE = "tmp/train_data.pickle"

dataset = TextDataset(vocabulary_size=int(opt.vocab_size),
                      min_occurrences=5,
                      window_size=int(opt.window_size),
                      name='GloveDataset',
                      text_dir=TEXT_DIR)

dataset.prepare()