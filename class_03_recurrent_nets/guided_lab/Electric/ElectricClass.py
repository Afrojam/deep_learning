"""
.. module:: Process

Process
*************

:Description: Process



:Authors: bejar


:Version:

:Created on: 31/08/2017 8:26

"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1
    impl = 1

    print("Starting:", time.ctime())

    ############################################
    # Data

    train = np.loadtxt('class_03_recurrent_nets/guided_lab/Electric/ElectricDevices_TRAIN.csv', delimiter=',')
    print(train.shape)
    train_x = train[:, 1:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    train_y = train[:, 0] - 1
    print(np.unique(train_y))
    nclasses = len(np.unique(train_y))
    train_y_c = np_utils.to_categorical(train_y, nclasses)
    test = np.loadtxt('class_03_recurrent_nets/guided_lab/Electric/ElectricDevices_TEST.csv', delimiter=',')

    print(test.shape)
    test_x = test[:, 1:]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    test_y = test[:, 0] - 1
    test_y_c = np_utils.to_categorical(test_y, nclasses)

    ############################################
    # Model

    RNN = GRU #LSTM  # GRU
    neurons = 64
    drop = 0.2
    nlayers = 4  # >= 1

    model = Sequential()

    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, recurrent_dropout=drop))
    else:
        model.add(
            RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, recurrent_dropout=drop, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))

        model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    ############################################
    # Training

    # optimizer = RMSprop(lr=0.01)
    optimizer = SGD(lr=0.01, momentum=0.95)
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = 50
    batch_size = 128
    history = model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_x, test_y_c), verbose=verbose)

    ############################################
    # Results

    score, acc = model.evaluate(test_x, test_y_c, batch_size=batch_size, verbose=0)

    print()
    print('ACC= ', acc)
    print()
    test_pred = model.predict_classes(test_x, verbose=0)
    print('Confusion Matrix')
    print('-'*20)
    print(confusion_matrix(test_y, test_pred))
    print()
    print('Classification Report')
    print('-'*40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())
