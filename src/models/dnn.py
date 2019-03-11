import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.backend import tensorflow_backend
from keras.models import Sequential, load_model
from keras.layers import Dense

BATCH_SIZE = 128
SHAPE = [128, 64, 64]

print(tensorflow_backend._get_available_gpus())

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
    dnn = Sequential()

    dnn.add(Dense(units=SHAPE[0], activation='relu', input_dim=X_train.shape[1]))
    for units_cnt in SHAPE[1:]:
        dnn.add(Dense(units=units_cnt, activation='relu'))
    dnn.add(Dense(units=1, activation='linear'))

    dnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    steps_cnt = X_train.shape[0]/BATCH_SIZE
    dnn.fit_generator(generator=nn_batch_generator(X_train, y_train,
                                                   steps_cnt),
                      nb_epoch=3, steps_per_epoch=steps_cnt)

    print("DNN train done")
    return dnn


def save(dnn, model_dirpath):
    dnn.save(os.path.join(model_dirpath, "dnn.h5"))


def load(model_dirpath):
    return load_model(os.path.join(model_dirpath, "dnn.h5"))


def predict(dnn, X_test):
    return dnn.predict(X_test)
