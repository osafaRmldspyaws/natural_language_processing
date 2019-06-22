# import the required libraries

import numpy as np
from bi_gram.markov import get_bigram_prob, get_sentences_with_word2idx
from datetime import datetime
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    #load the data
    sentences, word2idx = get_sentences_with_word2idx()

    #Vocan=b size
    V = len(sentences)
    print('The Vocab size : {}'.format(V))

    # we will also treat beginning and end of the sentence as bigrams
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    #get the prob for counts
    bigram_prob = get_bigram_prob(sentences, V, start_idx, end_idx, smoothing=0.1)

    #train a shallow neural netwrok
    D = 100
    W1 = np.random.randn((V, D))/ np.srqt(V)
    W2 = np.random.randn((D, V))/ np.sqrt(D)

    losses = []
    epochs = 1
    lr = 1e-1

    def softmax(a):
        a = a - max(a)
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    W_bigram = np.log(bigram_prob)
    bigram_losses = []

    t0 = datetime.now()
    for epoch in range(epochs):
        #shuffle the sentences
        random.shuffle(sentences)
        j = 0
        for sentence in sentences:
            # convert the sentences into one-hot encoded vectors(both inputs and targets)
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = np.zeros((n-1, V))
            targets = np.zeros((n-1, V))
            inputs[np.arange(n-1), sentence[: n-1]] =1
            targets[np.arange(n-1), sentence[: n-1]] =1

            #get output predictions
            hidden = np.tanh(inputs.dot(W1))
            predictions = softmax(hidden.dot(W2))

            # perform gradient descent
            W2 -= lr* hidden.T.dot(predictions - targets)
            dhidden = (predictions - hidden).dot(W2.T) * (1 - hidden*hidden)
            W1 -= lr* inputs.T.dot(dhidden)

            # keep the track of losses
            loss = -np.sum(targets * np.log(predictions)) / (n-1)
            losses.append(loss)
            



