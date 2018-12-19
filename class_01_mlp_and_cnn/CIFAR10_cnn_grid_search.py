from __future__ import division
from pathlib import Path
import time

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.models import model_from_json #to load a model from a json file.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import toimage
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

np.random.seed(12345678)

# global variables
# NN architecture
n_hlayers = 3 #more than 3 hidden layers shrinks the image too much. Try only convolutions without pooling. Play with strides and greater filters
initial_layer_neurons = 32
neurons = 200
layers = 6
n_epochs = 50
n_batch_size = 128
dropout_list = [0.0, 0.2, 0.4, 0.6]
dropout = 0.
kernel_initializer_list = ['glorot_normal', 'he_normal']
batch_normalization = False
neurons_list = np.power(2,range(1,11))
activation = "relu"


# Auxiliary lists
layers_accuracy = []
neurons_accuracy = []
kernel_accuracy = []
optimizer_accuracy = []
callback_list = []

# paths and file names
plots_folder = Path("plots/grid_search_cnn")
models_folder = Path("models/")
dataset_name = "cifar"
original_model_name = 'cnn_{}_grid_search_cnn'.format(dataset_name)
imgs_folder = Path("imgs/")
text_file = "CIFAR-CNN_grid_search_cnn_score_file.txt"
open(text_file, 'w+').close()


def show_img(X):
    plt.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            plt.subplot2grid((4, 4), (i, j))
            plt.imshow(toimage(X[k]))
            k = k + 1
    plt.savefig(str(imgs_folder / "cifar_images"))
    plt.close()


def do_plot(plot_name, plot_object, plot_title, plot_xlabel, plot_ylabel, plot_legend):
    for item in plot_object:
        plt.plot(item)
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend(plot_legend, loc='upper left')
    plt.savefig(str(plot_name))
    plt.close()


if __name__ == "__main__":
    start_time = time.clock()
    print("Using keras version {0}".format(keras.__version__))

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # loading of CIFAR dataset

    # check sizes
    print("\n")
    print("Number of training examples: '{0}'".format(x_train.shape[0]))
    print("Number of test examples: '{0}'".format(x_test.shape[0]))
    print("Size of train samples: '{0}'".format(x_train.shape[1:]))

    # Data to 1D and normalization
    # x_train = x_train.reshape(60000, 784)  # 60000 observations of 784 features
    # x_test = x_test.reshape(10000, 784)  # 10000 observations of 784 features

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    train_input_shape = x_train.reshape(*x_train.shape)
    test_input_shape = x_test.reshape(*x_test.shape)

    # Adapts labels to one hot encoding vector for softmax classifier
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # keras callbacks
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=5,
                                              verbose=0, mode='auto')
    callback_list = [earlystop]

    # Neural network architecture
    for (optimizer, optimizer_name) in zip([keras.optimizers.SGD(),
                                            keras.optimizers.Adam(),
                                            keras.optimizers.Adam(amsgrad=True),
                                            keras.optimizers.Nadam(),
                                            keras.optimizers.RMSprop(),
                                            keras.optimizers.Adamax()],
                                           ["SGD", "Adam_Default", "Adam_amsgrad",
                                            "Nadam", "RMSprop", "Adamax"]):
        kernel_accuracy = []
        for kernel_initializer in kernel_initializer_list:
            # for dropout in dropout_list:
            layers_accuracy = []
            for layers in range(n_hlayers):
                neurons_accuracy = []
                for neurons in neurons_list:
                    nn = Sequential()
                    nn.add(Conv2D(initial_layer_neurons, kernel_size=(3, 3), padding='same', strides=1, activation=activation, kernel_initializer=kernel_initializer, input_shape=x_train.shape[1:]))
                    nn.add(MaxPooling2D(pool_size=(2, 2)))
                    if dropout != 0.:
                        nn.add(Dropout(dropout))
                    if batch_normalization:
                        nn.add(BatchNormalization())
                    for i in range(layers):
                        nn.add(Conv2D(neurons, (3, 3), padding='same', strides=1, kernel_initializer=kernel_initializer, activation=activation))
                        nn.add(MaxPooling2D(pool_size=(2, 2)))
                        if dropout != 0.:
                            nn.add(Dropout(dropout))
                        if batch_normalization:
                            nn.add(BatchNormalization())

                    nn.add(Flatten())
                    nn.add(Dense(128, activation=activation))
                    nn.add(Dense(10, activation='softmax'))

                    # Compile the model
                    nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                    # composing the model name
                    model_name = original_model_name
                    model_name = model_name + '_{}_hidden_layers'.format(layers)
                    model_name = model_name + '_{}_neurons'.format(neurons)
                    model_name = model_name + '_' + kernel_initializer
                    model_name = model_name + '_{}_dropout'.format(dropout)
                    if batch_normalization:
                        model_name = model_name + '_batch_normalization'
                    model_name = model_name + '_' + optimizer_name

                    history = nn.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=n_batch_size,
                                     epochs=n_epochs, callbacks=callback_list)

                    # Evaluate the model
                    score = nn.evaluate(x_test, y_test, verbose=0)

                    # Store plots
                    # Accuracy plot
                    plot_name = plots_folder / 'model_accuracy_{0}.png'.format(model_name)
                    do_plot(plot_name, [history.history['acc'], history.history['val_acc']], 'model accuracy', 'epoch', 'accuracy', ['train', 'test'])
                    # Loss plot
                    plot_name = plots_folder / 'model_loss_{0}.png'.format(model_name)
                    do_plot(plot_name, [history.history['loss'], history.history['val_loss']], 'model loss',
                                                                                               'epoch', 'loss',
                            ['train', 'test'])

                    with open('{}'.format(text_file), 'a') as f:
                        f.write("Scores for neural network model {}: \n".format(model_name))
                        f.write("Loss {}".format(score[0]))
                        f.write("\n")
                        f.write("Accuracy {}".format(score[1]))
                        f.write("\n\n")
                    neurons_accuracy.append(score[1])
                # neurons plot
                do_plot(plots_folder / 'model_neurons_acc_{0}.png'.format(model_name), [neurons_accuracy],
                        'model neurons acc',
                        'number of neurons (power of 2)', 'accuracy', ['acc'])
                layers_accuracy.append(score[1])
            # Layers plot
            do_plot(plots_folder / 'model_layers_{0}.png'.format(model_name), [layers_accuracy], 'model layers acc',
                    'number of layers', 'accuracy', ['acc'])
            kernel_accuracy.append(score[1])
        # kernels plot
        do_plot(plots_folder / 'model_kernels_{0}.png'.format(model_name), [kernel_accuracy], 'model kernels acc',
                None, 'accuracy', ['acc'])
        optimizer_accuracy.append(score[1])
    # optimizer plot
    do_plot(plots_folder / 'model_optimizer_{0}.png'.format(model_name), [optimizer_accuracy], 'model optimizer acc',
            None, 'accuracy', ['acc'])

    # accuracy_dict[model_name: score[1]]

    # Confusion Matrix
    # Compute probabilities
    # Y_pred = nn.predict(x_test)
    # # Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # # Plot statistics
    # print('Analysis of results')
    # target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    # print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

    # Saving model and weights
    # nn_json = nn.to_json()
    # json_file_name = '{0}.json'.format(model_name)
    # with open(models_folder / json_file_name, 'w') as json_file:
    #     json_file.write(nn_json)
    # weights_file_name = "weights-{0}_".format(model_name) + str(score[1]) + ".hdf5"
    # weights_file = models_folder / weights_file_name
    # nn.save_weights(str(weights_file), overwrite=True)
    #
    # # Loading model and weights
    # json_file = open(models_folder / json_file_name, 'r')
    # nn_json = json_file.read()
    # json_file.close()
    # nn = model_from_json(nn_json)
    # nn.load_weights(weights_file)

    # measuring execution time
    print("Total execution time {} seconds".format(time.clock() - start_time))
