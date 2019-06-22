# import the required libraries

from bi_gram.markov import get_sentences_with_word2idx, get_bigram_prob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ =='__main__':

    #load the data
    sentences, word2idx = get_sentences_with_word2idx()

    #vocab size
    V = len(word2idx)
    print("Vocab size : {}".format(V))

    # we will treat beginning of the sentence and end of the sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where :
    # row = last word
    # column = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_prob(sentences,V, start_idx, end_idx, smoothing=0.1)

    #train a logistic model
    # intialize a weight matrix
    W = np.random.randn(V, V)/np.sqrt(V)

    losses = []
    epochs = 1
    lr = 1e-1

    def softmax(a):
        a = a - max(a)
        exp_a = np.exp(a)
        return exp_a/exp_a.sum(axis=1, keepdims=True)


    W_bigrams = np.log(bigram_probs)
    bigram_losses = []

    dat0 = datetime.now()
    for epoch in range(epochs):
        # shuffle the sentence at each epoch
        j=0
        for sentence in sentences:
            # convert the sentences into one-hot-encoded inputs and targets
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = np.zeros(((n-1), V))
            targets = np.zeros(((n-1), V))
            inputs[np.arange(n-1), sentence[:n-1]] = 1
            targets[np.arange(n-1), sentence[:n-1]] = 1

            # get the predictions
            predictions = softmax(inputs.dot(W))

            # perform the gradient descent
            W = W - (lr * inputs.T.dot(predictions - targets))

            #keep track of the loss
            loss = -np.sum(targets * np.log(predictions)) / (n-1)
            losses.append(loss)

            # keep track of bi-gram losses
            if epoch == 0:
                bigram_predictions = softmax(inputs.T.dot(W))
                bigram_loss = -np.sum(targets * np.log(bigram_predictions)) /(n-1)
                bigram_losses.append(bigram_loss)

            if j%10 == 0:
                print('epoch : ', epoch, 'sentence : ')

    print('Elapsed time to train : {} '.format(datetime.now() - dat0))
    plt.plot(losses)
    plt.show()






