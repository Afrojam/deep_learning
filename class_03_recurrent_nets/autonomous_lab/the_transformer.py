from __future__ import unicode_literals, print_function, division

import datetime
from io import open
import unicodedata
import string
import re
import random
import time
import math
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
teacher_forcing_ratio = 0.5
plot_number = 0
total_score = 0
model_name = "seq2seq_att"
results_dict = {}
path = ""


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print("Input language: {}, number of words: {}".format(input_lang.name, input_lang.n_words))
    print("Output language: {}, number of words: {}".format(output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # number of units of the RNN layer

        self.embedding = nn.Embedding(input_size,
                                      hidden_size)  # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.self_attention = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)  # Defining GRU layer

    def forward(self, input, hidden):  # 2 inputs: the input and the prev hidden state.
        embedded = self.embedding(input).view(1, 1, -1)  # take the word
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # number of hidden units in the RNN

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(
            self.out(output[0]))  # classification of the output word given input and hidden state of encoder
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)  # Embedding layer, transform inputs to embeds
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1)  # Vector concatenating the embedded and the hidden_state
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(
            0))  # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.

        output = torch.cat((embedded[0], attn_applied[0]),
                           1)  # Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        output = self.attn_combine(output).unsqueeze(
            0)  # Returns a new tensor with an added dimension to the original tensor.
        # unsqueeze = redimension adding a dimension
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    loss = 0

    ### INITIALIZING ENCODER
    encoder_hidden = encoder.initHidden()  # torch.zeros(1, 1, self.hidden_size, device=device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    ### TRAINING ENCODER
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)  # feed the encoder with a word and last hidden state
        encoder_outputs[ei] = encoder_output[
            0, 0]  # collect the output of the encoder with input tensor e(ncoder)i(nput).

    ### INITIALIZING DECODER
    decoder_input = torch.tensor([[SOS_token]], device=device)  # start sentence with SOS_token for Decoder
    decoder_hidden = encoder_hidden  # take encoder hidden state for input for the decoder.

    ### INITIALIZING TEACHER FORCING
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(
                1)  # Returns the k largest elements of the given input tensor along a given dimension.
            # because decoder_output is a vector of probs for each class, we want the top scoring class value and index.
            decoder_input = topi.squeeze().detach()  # detach from history as input, y.hat as x.

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:  # in case the output of the decoder was <EOS>, stop iterating and take another pair.
                break

    ### UPDATE PARAMETERS
    loss.backward()  # Computation of the loss after passing from the encoder to the decoder, so we update both after.
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length  # return the loss for the pair.


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(path + "/loss_plot_{}.png".format(datetime.datetime.now().strftime('%d%m%Y%H%s')))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]  # creates the train as random pair picks from the dataset
    criterion = nn.NLLLoss()  # negative log likelihood loss, for classification

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]  # select x and y pairs
        input_tensor = training_pair[0]  # select x
        target_tensor = training_pair[1]  # select y

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    results_dict['Loss'] = loss
    results_dict['Time'] = time.time() - start
    results_dict['Number of parameters'] = getModelParametersNumber(encoder) + getModelParametersNumber(decoder)
    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        ### READING SENTENCE
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        ### INITIALIZING ENCODER
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        ### FEEDING THE ENCODER
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        ### INITIALIZING DECODER
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        ## FEEDING DECODER
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')  # break if end of sentence as y.hat (output)
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])  # cain of output words

            decoder_input = topi.squeeze().detach()  # take output as input

        return decoded_words, decoder_attentions[:di + 1]  # returns outputs and the attention vectors


def evaluateRandomly(encoder, decoder, n=1000):
    cum_score = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        score = scoring(pair[1].split(" "), output_words)
        cum_score = cum_score + score
        print('<', output_sentence)
        print('Score = {}'.format(score))
        print('')
    avg_score = cum_score / n
    print("Average score of the model = {}".format(avg_score))
    results_dict["BLEU score"] = avg_score


def scoring(reference, candidate):
    score = sentence_bleu(reference, candidate)
    return score


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig(path + "/Attention_{}.png".format(input_sentence))


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def getModelParametersNumber(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def saveResultsToFile():
    with open(path + '/' + results_name, "w+") as results_file:
        for (value, key) in results_dict.items():
            print(value, key, file=results_file)
        print("\n\n", file=results_file)

if __name__ == "__main__":
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    path = model_name
    timestamp = datetime.datetime.now().strftime('%d%m%Y%H')
    results_name = "results_model_%s_%s.txt" % (model_name, timestamp)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

    evaluateRandomly(encoder1, attn_decoder1)

    saveResultsToFile()

    print("Visualizing attention weights: ")
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())

    evaluateAndShowAttention("elle a cinq ans de moins que moi .")

    evaluateAndShowAttention("elle est trop petit .")

    evaluateAndShowAttention("je ne crains pas de mourir .")

    evaluateAndShowAttention("c est un jeune directeur plein de talent .")
