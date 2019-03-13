import os
import numpy as np

from src import common

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf           # NOQA: E402
import tensorflow.keras as keras  # NOQA: E402

BATCH_SIZE = 128
DNN_SHAPE = [128, 64, 64]

print(tf.test.is_gpu_available())
tf.set_random_seed(common.RANDOM_SEED)
np.random.seed(common.RANDOM_SEED)


def nn_batch_generator(X_data, y_data, steps_cnt):
    batch_size = int(X_data.shape[0]/steps_cnt)
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while True:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = np.array(X_data[index_batch, :].todense())
        y_batch = y_data[index_batch]
        counter += 1
        assert(not np.isnan(X_batch).any())
        assert(not np.isinf(X_batch).any())
        yield X_batch, y_batch
        if (counter > steps_cnt):
            counter = 0


def train(X_train, y_train):
    dnn = keras.Sequential()

    dnn.add(keras.layers.Dense(units=DNN_SHAPE[0], activation=tf.nn.selu,
                               input_dim=X_train.shape[1]))
    for units_cnt in DNN_SHAPE[1:]:
        dnn.add(keras.layers.Dense(units=units_cnt, activation=tf.nn.selu))
    dnn.add(keras.layers.Dense(units=1, activation='linear'))

    dnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    steps_cnt = int(X_train.shape[0]/BATCH_SIZE)
    dnn.fit_generator(generator=nn_batch_generator(X_train, y_train,
                                                   steps_cnt),
                      epochs=3, steps_per_epoch=steps_cnt)

    print("DNN train done")
    return dnn


def save(dnn, model_dirpath):
    dnn.save(os.path.join(model_dirpath, "dnn.h5"))


def load(model_dirpath):
    return keras.models.load_model(os.path.join(model_dirpath, "dnn.h5"))


def predict(dnn, X_test):
    return dnn.predict(X_test)
