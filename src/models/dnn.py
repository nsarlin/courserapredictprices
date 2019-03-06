from keras.models import Sequential
from keras.layers import Dense
from keras.backend import tensorflow_backend
from keras.models import load_model

import os
import numpy as np

BATCH_SIZE = 256

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensors((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def nn_batch_generator(X_data, y_data, steps_cnt):
    batch_size = int(X_data.shape[0]/steps_cnt)
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while True:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = np.array(X_data[index_batch,:].todense())
        y_batch = y_data[index_batch]
        counter += 1
        assert(not np.isnan(X_batch).any())
        assert(not np.isinf(X_batch).any())
        yield X_batch, y_batch
        if (counter > steps_cnt):
            counter=0


def train(X_train, y_train):

    print(tensorflow_backend._get_available_gpus())
    dnn = Sequential()

    dnn.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
    dnn.add(Dense(units=64, activation='relu'))
    dnn.add(Dense(units=64, activation='relu'))
    dnn.add(Dense(units=1, activation='linear'))
    dnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    steps_cnt = X_train.shape[0]/BATCH_SIZE
    history = dnn.fit_generator(generator=nn_batch_generator(X_train, y_train, steps_cnt),
                                nb_epoch=3,
                                steps_per_epoch=steps_cnt)

    print("DNN train done")
    return dnn


def save(dnn, model_dirpath):
    dnn.save(os.path.join(model_dirpath, "dnn.h5"))


def load(model_dirpath):
    return load_model(os.path.join(model_dirpath, "dnn.h5"))
