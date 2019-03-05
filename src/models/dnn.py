from keras.models import Sequential
from keras.layers import Dense
from keras.backend import tensorflow_backend

import numpy as np


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensors((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
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
        if (counter > number_of_batches):
            counter=0


def train(X_train, y_train):

    print(tensorflow_backend._get_available_gpus())
    dnn = Sequential()

    dnn.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
    dnn.add(Dense(units=64, activation='relu'))
    dnn.add(Dense(units=64, activation='relu'))
    dnn.add(Dense(units=1, activation='linear'))
    dnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    batch_size = 64
    history = dnn.fit_generator(generator=nn_batch_generator(X_train, y_train, batch_size),
                                nb_epoch=3,
                                steps_per_epoch=X_train.shape[0]/batch_size)

    print("DNN train done")
    return dnn
