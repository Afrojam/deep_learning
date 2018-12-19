"""
.. module:: SentimentTwAir

SentimentTwAir
*************

:Description: SentimentTwAir


:Version: 

:Created on: 07/09/2017 9:05 

"""

import pandas
from sklearn.metrics import confusion_matrix, classification_report
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from collections import Counter
import argparse
import time


def tweet_to_words(raw_tweet):
    """
    Only keeps ascii characters in the tweet and discards @words

    :param raw_tweet:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", raw_tweet)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    print("Starting:", time.ctime())

    ############################################
    # Data

    Tweet = pandas.read_csv("Airlines.csv")
    Tweet = pandas.read_csv("Presidential.csv")

    # Pre-process the tweet and store in a separate column
    Tweet['clean_tweet'] = Tweet['text'].apply(lambda x: tweet_to_words(x))
    # Convert sentiment to binary
    Tweet['sentiment'] = Tweet['twsentiment'].apply(lambda x: 0 if x == 'negative' else 1 if x == 'positive' else 2)

    # Join all the words in review to build a corpus
    all_text = ' '.join(Tweet['clean_tweet'])
    words = all_text.split()

    # Convert words to integers
    counts = Counter(words)

    numwords = 5000  # Limit the number of words to use
    vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    tweet_ints = []
    for each in Tweet['clean_tweet']:
        tweet_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])

    # Create a list of labels
    labels = np.array(Tweet['sentiment'])

    # Find the number of tweets with zero length after the data pre-processing
    tweet_len = Counter([len(x) for x in tweet_ints])
    print("Zero-length reviews: {}".format(tweet_len[0]))
    print("Maximum tweet length: {}".format(max(tweet_len)))

    # Remove those tweets with zero length and its corresponding label
    tweet_idx = [idx for idx, tweet in enumerate(tweet_ints) if len(tweet) > 0]
    labels = labels[tweet_idx]
    Tweet = Tweet.iloc[tweet_idx]
    tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

    seq_len = max(tweet_len)
    features = np.zeros((len(tweet_ints), seq_len), dtype=int)
    for i, row in enumerate(tweet_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]

    split_frac = 0.8
    split_idx = int(len(features) * 0.8)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print("Train set: \t\t{}".format(train_y.shape),
          "\nValidation set: \t{}".format(val_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    ############################################
    # Model
    drop = 0.0
    nlayers = 2  # >= 1
    RNN = LSTM  # GRU

    neurons = 128
    embedding = 40

    model = Sequential()
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len))

    if nlayers == 1:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop))
    else:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    ############################################
    # Training

    learning_rate = 0.01
    optimizer = SGD(lr=learning_rate, momentum=0.95)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = 50
    batch_size = 100

    train_y_c = np_utils.to_categorical(train_y, 3)
    val_y_c = np_utils.to_categorical(val_y, 3)

    model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_x, val_y_c),
              verbose=verbose)

    ############################################
    # Results

    test_y_c = np_utils.to_categorical(test_y, 3)
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,
                                verbose=verbose)
    print()
    print('Test ACC=', acc)

    test_pred = model.predict_classes(test_x, verbose=verbose)

    print()
    print('Confusion Matrix')
    print('-'*20)
    print(confusion_matrix(test_y, test_pred))
    print()
    print('Classification Report')
    print('-'*40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())