from __future__ import division
from pathlib import Path
import time

import keras
import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.models import model_from_json #to load a model from a json file.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#global variables
plots_folder = Path("plots/")
models_folder = Path("models/")
model_name = 'nn_mnist'

if __name__ == "__main__":
    K.clear_session()
    start_time = time.clock()
    print("Using keras version %s" % keras.__version__)

    #Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # loading of MNIST dataset

    #check sizes
    print("\n")
    print("Number of training examples: '{0}'".format(x_train.shape[0]))
    print("Size of train samples: '{0}'".format(x_train.shape[1:]))
    print("Size of test samples: '{0}'".format(x_test.shape[1:]))

    #Data to 1D and normalization
    x_train = x_train.reshape(60000, 784) #60000 observations of 784 features
    x_test = x_test.reshape(10000, 784) # 10000 observations of 784 features

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # Adapts labels to one hot encoding vector for softmax classifier
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    #Neural network architecture
    # Two hidden layers
    nn = Sequential()
    nn.add(Dense(128, activation='relu', input_shape=(784,)))
    nn.add(Dense(64, activation = 'relu'))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(10, activation='softmax'))

    #Model visualization
    #The plot of the model needs pydot, graphviz and pydot-ng
    #plot_model(nn, to_file='nn.png', show_shapes = True)

    #Compile the model
    nn.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

    history = nn.fit(x_train, y_train, batch_size = 128, epochs=20)

    #Evaluate the model
    score = nn.evaluate(x_test, y_test, verbose=0)

    #Store plots
    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(str(plots_folder / '{}_accuracy.png'.format(model_name)))
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(str(plots_folder / '{}_loss.png'.format(model_name)))

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
    nn_json = nn.to_json()
    with open(models_folder / '{}.json'.format(model_name), 'w') as json_file:
        json_file.write(nn_json)
    weights_file_name = "weights-{}_".format(model_name) + str(score[1]) + ".hdf5"
    weights_file = models_folder / weights_file_name
    nn.save_weights(str(weights_file), overwrite=True)

    # Loading model and weights
    json_file = open(models_folder / '{0}.json'.format(model_name), 'r')
    nn_json = json_file.read()
    json_file.close()
    nn = model_from_json(nn_json)
    nn.load_weights(weights_file)

    #Execution time
    print("Total execution time {} seconds ".format(time.clock()-start_time))

    # Ejemplo Confunde el 0 con el 6 (7 errores), el 9 con el 4 (20 errores) y el 3 con el 8 (21 errores).
    #La NN tiene un 96% accuracy