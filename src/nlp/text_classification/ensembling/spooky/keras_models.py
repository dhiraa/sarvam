from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

from sklearn import preprocessing
import  numpy as np
from tqdm import tqdm

def simple_net(train_x_glove_scl, train_y_one_hot_encoded,
               val_x_glove_scl, val_y_one_hot_encoded):


    model = Sequential()

    model.add(Dense(300, input_dim=300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(train_x_glove_scl, y=train_y_one_hot_encoded, batch_size=64,
              epochs=5, verbose=1,
              validation_data=(val_x_glove_scl, val_y_one_hot_encoded))


def bilstm_net(train_x, train_y_one_hot_encoded,
               val_x, val_y_one_hot_encoded, embeddings_index):
    # using keras tokenizer here
    token = text.Tokenizer(num_words=None)
    max_len = 70

    token.fit_on_texts(list(train_x) + list(val_x))
    train_x_seq = token.texts_to_sequences(train_x)
    val_x_seq = token.texts_to_sequences(val_x)

    # zero pad the sequences
    train_x_pad = sequence.pad_sequences(train_x_seq, maxlen=max_len)
    val_x_pad = sequence.pad_sequences(val_x_seq, maxlen=max_len)

    word_index = token.word_index

    # create an embedding matrix for the words we have in the dataset
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # A simple LSTM with glove embeddings and two dense layers
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        300,
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Fit the model with early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model.fit(train_x_pad, y=train_y_one_hot_encoded,
              batch_size=64, epochs=100,
              verbose=1,
              validation_data=(val_x_pad, val_y_one_hot_encoded),
              callbacks=[earlystop])





