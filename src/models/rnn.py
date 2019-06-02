import os
import numpy as np
from itertools import product

from src import common

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf           # NOQA: E402
import tensorflow.keras as keras  # NOQA: E402

BATCH_SIZE = 32
RNN_SHAPE = [50]
N_STEPS = 5

print(tf.test.is_gpu_available())
tf.set_random_seed(common.RANDOM_SEED)
np.random.seed(common.RANDOM_SEED)

def rnn_series_batch_generator(X_data, y_data, batch_cnt):
    # TODO
    print("shape")
    print(X_data.shape)
    print("max")
    print(np.amax(X_data[:,0]))
    exit()

def train(X_train, y_train):
    rnn = keras.Sequential()
    # TODO: remove lags
    rnn.add(keras.layers.LSTM(units=RNN_SHAPE[0], activation=tf.nn.selu,
                              input_shape=(N_STEPS, X_train.shape[1])))
    rnn.add(keras.layers.Dense(units=1, activation='linear'))

    rnn.compile(loss='mean_squared_error', optimizer=keras.optimizers.Nadam(0.01), metrics=['mae'])
    batch_cnt = int(X_train.shape[0]/(BATCH_SIZE*N_STEPS))
    rnn.fit_generator(generator=rnn_series_batch_generator(X_train, y_train, batch_cnt),
                      epochs=3, steps_per_epoch=batch_cnt)

    return rnn


def save(rnn, model_dirpath):
    rnn.save(os.path.join(model_dirpath, "rnn.h5"))


def load(model_dirpath):
    return keras.models.load_model(os.path.join(model_dirpath, "rnn.h5"))

def prepare(X_train, labels):
    lag_col_names = ["{}-{}".format(colname, lagval) for colname, lagval \
                     in product(common.LAG_COLS, common.LAGS_LIST)]
    idx_tokeep = [idx for idx, val in enumerate(labels) if val not in lag_col_names]
    X_train = X_train[:,idx_tokeep]
    return X_train


def predict(rnn, X_test):
    return rnn.predict(X_test)
