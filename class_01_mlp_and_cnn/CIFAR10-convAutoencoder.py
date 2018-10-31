from __future__ import division
from pathlib import Path
import time

import keras
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
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

# paths and file names
plots_folder = Path("plots/autoencoder_cnn")
models_folder = Path("models/")
dataset_name = "cifar"
original_model_name = 'cnn_{}_autoencoder_cnn'.format(dataset_name)
imgs_folder = Path("imgs/")
text_file = "CIFAR-autoencoder_cnn_score_file.txt"
open(text_file, 'w+').close()

def show_img(X, title):
    plt.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            plt.subplot2grid((4, 4), (i, j))
            plt.imshow(toimage(X[k]))
            k = k + 1
    plt.savefig(str(imgs_folder / title))
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
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # showing some example images
    print("\n")
    print("Saving some images from the dataset in imgs folder")
    index = np.random.randint(len(x_test), size=16)
    selectedImages = x_test[index]
    show_img(selectedImages, 'original cifar10 images')

    # check sizes
    print("\n")
    print("Number of training examples: '{0}'".format(x_train.shape[0]))
    print("Number of test examples: '{0}'".format(x_test.shape[0]))
    print("Size of train samples: '{0}'".format(x_train.shape[1:]))
    # Category representation
    print("Category representation in train data: {}".format(np.unique(y_train, return_counts=True)))
    print("Category representation in test data: {}".format(np.unique(y_test, return_counts=True)))

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

    autoencoder = Sequential()
    autoencoder.add(Conv2D(32, (3,3), strides = 1, padding = 'same', activation = 'relu', input_shape=x_train.shape[1:]))
    autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
    autoencoder.add(Conv2D(16, (3, 3), strides=1, padding='same', activation='relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(8, (3, 3), strides=1, padding='same', activation='relu'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    #Encoder output layers
    print("Encoder outputs shape:")
    for layer in autoencoder.layers:
        print(layer.output_shape)

    #visualization pruposes
    autoencoder.add(Flatten())
    autoencoder.add(Reshape((4,4,8)))

    autoencoder.add(Conv2D(8, (3,3), strides = 1, padding = 'same', activation = 'relu'))
    autoencoder.add(UpSampling2D((2,2)))
    autoencoder.add(Conv2D(16, (3, 3), strides=1, padding='same', activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(3, (3,3), padding='same', activation='sigmoid'))
    #Decoder outputs
    print("Full outputs shape")
    for layer in autoencoder.layers:
        print(layer.output_shape)

    autoencoder.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    autoencoder.summary()

    encoder = Model(inputs = autoencoder.inputs, outputs= autoencoder.get_layer('flatten_1').output)
    #encoder.summary()

    history = autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), batch_size=n_batch_size,
                     epochs=n_epochs)

    reconstructed = autoencoder.predict(x_test[index])

    selectedImages = reconstructed
    show_img(selectedImages, 'reconstructed cifar10 images')