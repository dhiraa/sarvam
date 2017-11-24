from tqdm import tqdm

def vocab_to_tsv(vocab_list, outfilename):
    from tensorflow.python.platform import gfile
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

def tsv_to_vocab(filename):
    '''
    Open the vocab file and reads all lines and make dict for ID <-> Word conversion
    :param filename: 
    :return: 
    '''
    with open(filename) as file:
        lines = file.readlines()
    vocab = list(map(lambda line: line.strip(), lines))
    word_2_id = {word: i for i, word in enumerate(vocab)}
    id_2_word = {i: word for i, word in enumerate(vocab)}

    return word_2_id, id_2_word

