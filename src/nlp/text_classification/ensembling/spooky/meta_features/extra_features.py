import pandas as pd

from tqdm import tqdm

tqdm.pandas()

punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def clean_text(x):
    x.lower()
    for p in punctuation:
        x.replace(p, '')
    return x


def extract_features(df, train_flag=False):
    df['text_cleaned'] = df['text'].apply(lambda x: clean_text(x))
    df['n_.'] = df['text'].str.count('\.')
    df['n_...'] = df['text'].str.count('\...')
    df['n_,'] = df['text'].str.count('\,')
    df['n_:'] = df['text'].str.count('\:')
    df['n_;'] = df['text'].str.count('\;')
    df['n_-'] = df['text'].str.count('\-')
    df['n_?'] = df['text'].str.count('\?')
    df['n_!'] = df['text'].str.count('\!')
    df['n_\''] = df['text'].str.count('\'')
    df['n_"'] = df['text'].str.count('\"')

    # First words in a sentence
    df['n_The '] = df['text'].str.count('The ')
    df['n_I '] = df['text'].str.count('I ')
    df['n_It '] = df['text'].str.count('It ')
    df['n_He '] = df['text'].str.count('He ')
    df['n_Me '] = df['text'].str.count('Me ')
    df['n_She '] = df['text'].str.count('She ')
    df['n_We '] = df['text'].str.count('We ')
    df['n_They '] = df['text'].str.count('They ')
    df['n_You '] = df['text'].str.count('You ')
    df['n_the'] = df['text_cleaned'].str.count('the ')
    df['n_ a '] = df['text_cleaned'].str.count(' a ')
    df['n_appear'] = df['text_cleaned'].str.count('appear')
    df['n_little'] = df['text_cleaned'].str.count('little')
    df['n_was '] = df['text_cleaned'].str.count('was ')
    df['n_one '] = df['text_cleaned'].str.count('one ')
    df['n_two '] = df['text_cleaned'].str.count('two ')
    df['n_three '] = df['text_cleaned'].str.count('three ')
    df['n_ten '] = df['text_cleaned'].str.count('ten ')
    df['n_is '] = df['text_cleaned'].str.count('is ')
    df['n_are '] = df['text_cleaned'].str.count('are ')
    df['n_ed'] = df['text_cleaned'].str.count('ed ')
    df['n_however'] = df['text_cleaned'].str.count('however')
    df['n_ to '] = df['text_cleaned'].str.count(' to ')
    df['n_into'] = df['text_cleaned'].str.count('into')
    df['n_about '] = df['text_cleaned'].str.count('about ')
    df['n_th'] = df['text_cleaned'].str.count('th')
    df['n_er'] = df['text_cleaned'].str.count('er')
    df['n_ex'] = df['text_cleaned'].str.count('ex')
    df['n_an '] = df['text_cleaned'].str.count('an ')
    df['n_ground'] = df['text_cleaned'].str.count('ground')
    df['n_any'] = df['text_cleaned'].str.count('any')
    df['n_silence'] = df['text_cleaned'].str.count('silence')
    df['n_wall'] = df['text_cleaned'].str.count('wall')

    new_df = df.copy()
    # Find numbers of different combinations
    for c in tqdm(alphabet.upper()):
        new_df['n_' + c] = new_df['text'].str.count(c)
        new_df['n_' + c + '.'] = new_df['text'].str.count(c + '\.')
        new_df['n_' + c + ','] = new_df['text'].str.count(c + '\,')

        for c2 in alphabet:
            new_df['n_' + c + c2] = new_df['text'].str.count(c + c2)
            new_df['n_' + c + c2 + '.'] = new_df['text'].str.count(c + c2 + '\.')
            new_df['n_' + c + c2 + ','] = new_df['text'].str.count(c + c2 + '\,')

    for c in tqdm(alphabet):
        new_df['n_' + c + '.'] = new_df['text'].str.count(c + '\.')
        new_df['n_' + c + ','] = new_df['text'].str.count(c + '\,')
        new_df['n_' + c + '?'] = new_df['text'].str.count(c + '\?')
        new_df['n_' + c + ';'] = new_df['text'].str.count(c + '\;')
        new_df['n_' + c + ':'] = new_df['text'].str.count(c + '\:')

        for c2 in alphabet:
            new_df['n_' + c + c2 + '.'] = new_df['text'].str.count(c + c2 + '\.')
            new_df['n_' + c + c2 + ','] = new_df['text'].str.count(c + c2 + '\,')
            new_df['n_' + c + c2 + '?'] = new_df['text'].str.count(c + c2 + '\?')
            new_df['n_' + c + c2 + ';'] = new_df['text'].str.count(c + c2 + '\;')
            new_df['n_' + c + c2 + ':'] = new_df['text'].str.count(c + c2 + '\:')
            new_df['n_' + c + ', ' + c2] = new_df['text'].str.count(c + '\, ' + c2)

    # And now starting processing of cleaned text
    for c in tqdm(alphabet):
        new_df['n_' + c] = new_df['text_cleaned'].str.count(c)
        new_df['n_' + c + ' '] = new_df['text_cleaned'].str.count(c + ' ')
        new_df['n_' + ' ' + c] = new_df['text_cleaned'].str.count(' ' + c)

        for c2 in alphabet:
            new_df['n_' + c + c2] = new_df['text_cleaned'].str.count(c + c2)
            new_df['n_' + c + c2 + ' '] = new_df['text_cleaned'].str.count(c + c2 + ' ')
            new_df['n_' + ' ' + c + c2] = new_df['text_cleaned'].str.count(' ' + c + c2)
            new_df['n_' + c + ' ' + c2] = new_df['text_cleaned'].str.count(c + ' ' + c2)

            for c3 in alphabet:
                new_df['n_' + c + c2 + c3] = new_df['text_cleaned'].str.count(c + c2 + c3)

    if train_flag:
        new_df.drop(['text_cleaned', 'text', 'author', 'id'], axis=1, inplace=True)
    else:
        new_df.drop(['text_cleaned', 'text', 'id'], axis=1, inplace=True)

    df.drop(['text_cleaned'], axis=1, inplace=True)
    return new_df