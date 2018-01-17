import spacy
import time
from tqdm import tqdm
tqdm.pandas()

try:
    if spacy_nlp != None:
        spacy_nlp = spacy.load("en_core_web_sm")
except:
    spacy_nlp = spacy.load("en_core_web_sm")

def get_spacy_text(s):
    pos,tag,dep = '','',''
    for token in spacy_nlp(s):
        pos = pos + ' ' + token.pos_
        tag = tag + ' ' + token.tag_
        dep = dep + ' ' + token.dep_

    return pos,tag,dep


def get_spacy_features(df, text_col):
    start_t = time.time()
    poss, tags, deps = [], [], []
    for s in tqdm(df[text_col].values):
        pos, tag, dep = get_spacy_text(s)
        poss.append(pos)
        tags.append(tag)
        deps.append(dep)
    df['pos_txt'], df['tag_txt'], df['dep_txt'] = poss, tags, deps
    print('done creating spacy features', time.time() - start_t)

    return df[['pos_txt', 'tag_txt', 'dep_txt']]