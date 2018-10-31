from __future__ import division
from pathlib import Path
import time

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import model_from_json #to load a model from a json file.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import toimage
# from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
np.random.seed(12345678)



#NN architecture
n_hlayers = 5
n_neurons = 401
initial_neurons = 128
neurons = 200
# layers = 10
neurons_step = 20
n_epochs= 20
n_batch_size= 128
dropout_list = [0.0, 0.2, 0.4, 0.6]
dropout = 0.
kernel_initializer_list = ['random_normal', 'glorot_normal', 'he_normal']
kernel_initializer = 'glorot_normal'

#global variables
plots_folder = Path("plots/fine_tune_cnn")
models_folder = Path("models/")
dataset_name = "cifar"
original_model_name = 'cnn_{}'.format(dataset_name)
imgs_folder = Path("imgs/")
cifar_fnn_total_history = []
text_file = "CIFAR-CNN score file.txt"

layers_accuracy = []
neurons_accuracy = []
kernel_accuracy = []
optimizer_accuracy = []

if __name__=="__main__":
    start_time = time.clock()
    print("Using keras version {0}".format(keras.__version__))

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # loading of MNIST dataset

    # check sizes
    print("\n")
    print("Number of training examples: '{0}'".format(x_train.shape[0]))
    print("Number of test examples: '{0}'".format(x_test.shape[0]))
    print("Size of train samples: '{0}'".format(x_train.shape[1:]))

    # Data to 1D and normalization
    #x_train = x_train.reshape(60000, 784)  # 60000 observations of 784 features
    #x_test = x_test.reshape(10000, 784)  # 10000 observations of 784 features

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    train_input_shape = x_train.reshape(*x_train.shape)
    test_input_shape = x_test.reshape(*x_test.shape)

    # Adapts labels to one hot encoding vector for softmax classifier
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Neural network architecture
    # CNN
    nn = Sequential()
    nn.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=x_train.shape[1:]))
    nn.add(MaxPooling2D(pool_size=(2,2)))
    nn.add(Conv2D(64,(3,3), activation = 'relu'))
    nn.add(MaxPooling2D(pool_size=(2,2)))
    nn.add(Flatten())
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(10, activation='softmax'))

    # Model visualization
    # The plot of the model needs pydot, graphviz and pydot-ng
    #plot_model(nn, to_file='nn.png', show_shapes=True)

    # Compile the model
    for (optimizer, optimizer_name) in zip([keras.optimizers.SGD(),
                                            keras.optimizers.Adam(),
                 keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.995, epsilon=1e-08, decay=0.0005),
                 keras.optimizers.Adam(amsgrad= True),
                 keras.optimizers.Adagrad(),
                 keras.optimizers.Nadam(),
                 keras.optimizers.RMSprop(),
                 keras.optimizers.Adamax()],
                                           ["SGD","Adam_Default", "Adam_custom", "Adam_amsgrad", "Adagrad",
                                            "Nadam", "RMSprop", "Adamax"]):

        nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model_name = original_model_name
        model_name = model_name + optimizer_name

        history = nn.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=n_batch_size, epochs=n_epochs,
                         kernel_initializer=kernel_initializer)

        # Evaluate the model
        score = nn.evaluate(x_test, y_test, verbose=0)

        # Store plots
        # Accuracy plot
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plot_name = plots_folder /'model_accuracy_{0}'.format(model_name)
        plt.savefig(str(plot_name))
        plt.close()
        # Loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plot_name = plots_folder / 'model_loss_{0}'.format(model_name)
        plt.savefig(str(plot_name))
        plt.close()

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
        #
        # # Saving model and weights
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

    #measuring execution time
    print("Total execution time {} seconds".format(time.clock() - start_time))