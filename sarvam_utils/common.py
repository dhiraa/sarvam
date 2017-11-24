from tqdm import tqdm
from tensorflow.python.platform import gfile
import tensorflow as tf

def vocab_to_tsv(vocab_list, outfilename):
    '''

    :param vocab_list:
    :return:
    '''
    with gfile.Open(outfilename, 'wb') as file:
        for word in tqdm(vocab_list):
            if len(word) > 0:
                file.write("{}\n".format(word))

    nwords = len(vocab_list)
    print('{} words into {}'.format(nwords, outfilename))

    return nwords