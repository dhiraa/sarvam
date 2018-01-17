import sys
sys.path.append("../")
sys.path.append("../../")
from kaggle.dataset.spooky import SpookyDataset

from ensembling.spooky.tfidf import tfidf

# Use the sarvam utils to load the data set and get train/val/test data
dataset = SpookyDataset()
dataset.prepare()
dataframe = dataset.dataframe

train_x = dataframe.train_df
train_Y = dataframe.get_train_label()

val_x = dataframe.val_df
val_y = dataframe.get_val_label()

text_x = dataframe.test_df


train_x_tfidf, val_x_tfidf = tfidf(train_x, val_x)



