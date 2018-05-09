from __future__ import print_function
import sys
import numpy as np
import h5py
import scipy.io

import theano

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from methods import read_fasta, seq_matrix, seq_length
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from pandas.util.testing import K
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
import Bio, math
from sklearn.preprocessing import StandardScaler



from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from hyperas import optim
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from hyperas.distributions import choice, uniform, conditional



print('Loading Positive and negative data...')

def data():

    with open('data/ecoli.fa') as fp:

        pos_seq = []
        for name, seq in read_fasta(fp):
            pos_seq.append(seq)
        print('pos_seq: ', len(pos_seq))

        with open('data/ecoli_non.fa') as fp:

            neg_seq = []
            for name, seq in read_fasta(fp):
                neg_seq.append(seq)

        print('neg_seq: ', len(neg_seq))

        to_train1 = int(len(pos_seq) * 0.8)
        to_train2 = int(len(neg_seq) * 0.8)

        batch_size = 16
        to_train1 -= to_train1 % batch_size
        to_train2 -= to_train1 % batch_size

        pos_seq_train = pos_seq[:to_train1]
        neg_seq_train = neg_seq[:to_train2]
        pos_seq_test = pos_seq[to_train1:]
        neg_seq_test = neg_seq[to_train2:]
        print(str(len(pos_seq_train)) + ' positive train data loaded...')
        print(str(len(neg_seq_train)) + ' negative train data loaded...')

        pos_train_X, pos_train_y = seq_matrix(seq_list=pos_seq_train, label=1)
        seq_len = seq_length(pos_seq_train)
        neg_train_X, neg_train_y = seq_matrix(seq_list=neg_seq_train, label=0)

        X_train = np.concatenate((pos_train_X, neg_train_X), axis=0)
        y_train = np.concatenate((pos_train_y, neg_train_y), axis=0)

        print(str(len(pos_seq_test)) + ' positive test data loaded...')
        print(str(len(neg_seq_test)) + ' negative test data loaded...')

        pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test, label=1)
        neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test, label=0)

        X_test = np.concatenate((pos_test_X, neg_test_X), axis=0)
        y_test = np.concatenate((pos_test_y, neg_test_y), axis=0)
        return X_train, y_train, X_test, y_test

def length_seq(pos_seq_train):
    seq_list = pos_seq_train
    for i in range(len(seq_list)):
        seq = seq_list[i]

    length = len(seq)
    return length
print ('Building model...')

def model_auto(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Convolution1D(filters={{choice(50, 100, 150, 200, 250, 300)}},
                        kernel_size={{choice(7, 15, 21, 17)}},
                        input_shape=(length_seq, 4),
                        padding = 'valid'
                        ))

    model.add(MaxPooling1D(pool_size={{choice(2, 4, 6, 8, 10, 12)}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice(16, 32, 64, 128, 256)}}, activation ={{choice(['sigmoid', 'relu'])}}))
    model.add(Flatten())
    model.add(Dense({{choice(16, 32, 64, 128, 256)}}, activation ={{choice(['sigmoid', 'relu'])}}))

    print ('Compiling model...')
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
              loss='binary_crossentropy',
              metrics=['accuracy'])


    model.fit(X_train, y_train, validation_split=0.10, batch_size=16,epochs=10,shuffle = True, verbose=1)
    score = model.evaluate(X_test, y_test, batch_size=16)
    return score


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model_auto,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials()
                                              )
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
