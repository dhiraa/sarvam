import numpy as np
import pandas as pd
from sklearn import *

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
sub1 = pd.read_csv('../input/keras-bidirectional-lstm-baseline-lb-0-051/baseline.csv')

coly = [c for c in train.columns if c not in ['id','comment_text']]
y = train[coly]
tid = test['id'].values

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=50000)
data = tfidf.fit_transform(df)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(data[:nrow], y)
print(1- model.score(data[:nrow], y))
sub2 = pd.DataFrame(model.predict(data[nrow:]))
sub2.columns = coly
sub2['id'] = tid

sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    if c != 'id':
        blend[c] = blend[c] * 0.9 + blend[c+'_'] * 0.1
        blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
#blend.to_csv('submission.csv', index=False)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(data[:nrow+int(len(test)/2)], np.concatenate((y,blend[coly][:int(len(test)/2)].values), axis=0))
print(1- model.score(data[:nrow], y))
sub3 = np.concatenate((blend[coly][:int(len(test)/2)].values,model.predict(data[nrow+int(len(test)/2):])), axis=0)
sub3 = pd.DataFrame(sub3)
sub3.columns = coly
sub3['id'] = tid
sub3.to_csv('submission.csv', index=False)

sub2['id'] = tid