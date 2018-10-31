from __future__ import division
from pathlib import Path
import time

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import model_from_json #to load a model from a json file.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

np.random.seed(12345678)

# global variables
# NN architecture
n_hlayers = 3 #more than 3 hidden layers shrinks the image too much. Try only convolutions without pooling. Play with strides and greater filters
initial_layer_neurons = 32
neurons = 512
layers = 3
n_epochs = 100
n_batch_size = 128
dropout_list = [0.0, 0.2, 0.4, 0.6]
dropout = 0.
kernel_initializer_list = ['he_normal']
batch_normalization_list = [False, True]
neurons_list = np.power(2,range(1,11))
activation = "relu"
data_augmentation_list = [False, True]

# paths and file names
plots_folder = Path("plots/fine_tune_cnn")
models_folder = Path("models/")
dataset_name = "cifar"
original_model_name = 'cnn_{}_fine_tuned'.format(dataset_name)
imgs_folder = Path("imgs/")
text_file = "CIFAR-CNN_fine_tune_score_file.txt"
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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    train_input_shape = x_train.reshape(*x_train.shape)
    test_input_shape = x_test.reshape(*x_test.shape)

    # Adapts labels to one hot encoding vector for softmax classifier
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    #Concatenate data for new redimension
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.16666)
    sss.get_n_splits(x, y)
    # for train_index, test_index in sss.split(x, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]



    # keras callbacks
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=10,
                                              verbose=0, mode='auto')
    callback_list = [earlystop]



    # Neural network architecture
    for (optimizer, optimizer_name) in zip([keras.optimizers.Adamax()],
                                           ["Adamax"]):
        for kernel_initializer in kernel_initializer_list:
            for data_augmentation in data_augmentation_list:
                # Data augmentation
                if data_augmentation:
                    datagen = ImageDataGenerator(
                        featurewise_center=False,  # Set input mean to 0 over the dataset, feature-wise.
                        samplewise_center=False,  # Set each sample mean to 0.
                        featurewise_std_normalization=False,  # Divide inputs by std of the dataset, feature-wise.
                        samplewise_std_normalization=False,  # Divide each input by its std.
                        zca_whitening=False,  # ZCA whitening. WIth use of zca_epislon
                        rotation_range=90,  # Degree range for random rotations.
                        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                        horizontal_flip=True,  # randomly flip images
                        vertical_flip=False)
                    datagen.fit(x_train)
                # it could be good to save and see the images.
                for batch_normalization in batch_normalization_list:
                    for dropout in dropout_list:
                        nn = Sequential()
                        nn.add(Conv2D(initial_layer_neurons, kernel_size=(3, 3), padding='same', strides=1, activation=activation, input_shape=x_train.shape[1:]))
                        nn.add(MaxPooling2D(pool_size=(2, 2)))
                        if dropout != 0.:
                            nn.add(Dropout(dropout))
                        if batch_normalization:
                            nn.add(BatchNormalization())
                        for i in range(layers):
                            nn.add(Conv2D(neurons, (3, 3), padding='same', strides=1, activation=activation))
                            nn.add(MaxPooling2D(pool_size=(2, 2)))
                            if dropout != 0.:
                                nn.add(Dropout(dropout))
                            if batch_normalization:
                                nn.add(BatchNormalization())

                        nn.add(Flatten())
                        nn.add(Dense(128, activation=activation)) #1x1 convolution?
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
                        if data_augmentation:
                            model_name = model_name + '_data_augmentation'

                        if data_augmentation:
                            history = nn.fit_generator(datagen.flow(x_train, y_train, batch_size=n_batch_size), validation_data=(x_test, y_test),
                                         steps_per_epoch=len(x_train), epochs=n_epochs, callbacks=callback_list)
                        else:
                            history = nn.fit(x_train, y_train, validation_data=(x_test, y_test),
                                             batch_size=n_batch_size,
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
                                'epoch', 'loss', ['train', 'test'])

                        with open('{}'.format(text_file), 'a') as f:
                            f.write("Scores for neural network model {}: \n".format(model_name))
                            f.write("Loss {}".format(score[0]))
                            f.write("\n")
                            f.write("Accuracy {}".format(score[1]))
                            f.write("\n\n")

    # accuracy_dict[model_name: score[1]]

    # Confusion Matrix
    # Compute probabilities
    Y_pred = nn.predict(x_test)
    # Assign most probable label
    y_pred = np.argmax(Y_pred, axis=1)
    # Plot statistics
    print('Analysis of results')
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

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
