from keras.layers import Embedding, Dense, Flatten, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import gc
import pandas as pd
import numpy as np


def get_cnn_feats(train_df, test_df, text_col, category_col, rnd=42):
    num_of_train_samples = train_df.count()[0]
    num_of_test_samples = test_df.count()[0]

    FEAT_CNT = 5
    NUM_WORDS = 30000
    WORD_DIM = 10
    MAX_LEN = 100
    MODEL_P = '/tmp/nn_model.h5'

    tmp_x = train_df[text_col]
    tmp_y = train_df[category_col]
    tmp_x_test = test_df[text_col]

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(tmp_x)

    train_x_tokenized = tokenizer.texts_to_sequences(tmp_x)
    train_x_tokenized = pad_sequences(train_x_tokenized, maxlen=MAX_LEN)

    test_x_tokenized = tokenizer.texts_to_sequences(tmp_x_test)
    test_x_tokenized = pad_sequences(test_x_tokenized, maxlen=MAX_LEN)

    lb = preprocessing.LabelBinarizer()
    lb.fit(tmp_y)

    NUM_CLASSES = len(lb.classes_)

    ttrain_y = lb.transform(tmp_y)

    # return train pred prob and test pred prob
    train_pred, test_pred = np.zeros((num_of_train_samples, NUM_CLASSES)), \
                            np.zeros((num_of_test_samples, NUM_CLASSES))

    best_val_train_pred, best_val_test_pred = np.zeros((num_of_train_samples, NUM_CLASSES)), \
                                              np.zeros((num_of_test_samples, NUM_CLASSES))

    kf = KFold(n_splits=FEAT_CNT, shuffle=True, random_state=233 * rnd)

    for train_index, val_index in kf.split(tmp_x):
        # prepare aug train, val
        model = Sequential()

        model.add(Embedding(NUM_WORDS, WORD_DIM, input_length=MAX_LEN))

        model.add(GlobalAveragePooling1D())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_loss', save_best_only=True, verbose=1)

        model.fit(train_x_tokenized[train_index], ttrain_y[train_index],
                  validation_split=0.1,
                  batch_size=32, epochs=10,
                  verbose=2,
                  callbacks=[model_chk],
                  shuffle=False
                  )

        # save feat
        train_pred[val_index] = model.predict(train_x_tokenized[val_index])
        test_pred += model.predict(test_x_tokenized) / FEAT_CNT

        # best val model
        model = load_model(MODEL_P)
        best_val_train_pred[val_index] = model.predict(train_x_tokenized[val_index])
        best_val_test_pred += model.predict(test_x_tokenized) / FEAT_CNT

        # release
        del model
        gc.collect()
        print('------------------')

    return train_pred, test_pred, best_val_train_pred, best_val_test_pred