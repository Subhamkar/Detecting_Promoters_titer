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
from methods import read_fasta, seq_matrix, seq_length, length_seq 
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from pandas.util.testing import K
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
#import Bio, math
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
   
    with open('data/human_nontata.fa') as fp:
    #with open('data/ecoli.fa') as fp:
    #with open('data/bacillus.fa') as fp: 
        pos_seq = []
        for name, seq in read_fasta(fp):
            pos_seq.append(seq)
    print('pos_seq: ', len(pos_seq))


    with open('data/human_nontata_non.fa') as fp:
    #with open('data/ecoli_non.fa') as fp:
    #with open('data/bacillus_non.fa') as fp:
        neg_seq = []
        for name, seq in read_fasta(fp):
            neg_seq.append(seq)

    print('neg_seq: ', len(neg_seq))

    to_train1 = int(len(pos_seq)* 0.8)
    to_train2 = int(len(neg_seq)* 0.8)
    
    batch_size =16
    to_train1 -= to_train1 % batch_size
    to_train2 -= to_train1 % batch_size

    pos_seq_train = pos_seq[:to_train1]
    neg_seq_train = neg_seq[:to_train2]
    pos_seq_test = pos_seq[to_train1:]
    neg_seq_test = neg_seq[to_train2:]


   
    to_train3 = int(len(pos_seq_train)* 0.8)
    to_train4 = int(len(neg_seq_train)* 0.8)
    # tweak to match with batch_size
    batch_size =16
    to_train3 -= to_train3 % batch_size
    to_train4 -= to_train4 % batch_size

    pos_seq_val = pos_seq_train[to_train3:]
    neg_seq_val = neg_seq_train[to_train4:]
    pos_seq_train = pos_seq_train[:to_train3]
    neg_seq_train = neg_seq_train[:to_train4]
    


    print (str(len(pos_seq_train)) + ' positive train data loaded...')
    print (str(len(neg_seq_train)) + ' negative train data loaded...')

    pos_train_X, pos_train_y = seq_matrix(seq_list=pos_seq_train, label=1)
    #seq_len = seq_length(pos_seq_train)
    neg_train_X, neg_train_y = seq_matrix(seq_list=neg_seq_train, label=0)

    X_train = np.concatenate((pos_train_X, neg_train_X), axis=0)
    y_train = np.concatenate((pos_train_y, neg_train_y), axis=0)

    print (str(len(pos_seq_test)) + ' positive test data loaded...')
    print (str(len(neg_seq_test)) + ' negative test data loaded...')

    pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test, label=1)
    neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test, label=0)

    X_test = np.concatenate((pos_test_X, neg_test_X), axis=0)
    y_test = np.concatenate((pos_test_y, neg_test_y), axis=0)

    print (str(len(pos_seq_val)) + ' positive validation data loaded...')
    print (str(len(neg_seq_val)) + ' negative validation data loaded...')

    pos_val_X, pos_val_y = seq_matrix(seq_list=pos_seq_val, label=1)
    neg_val_X, neg_val_y = seq_matrix(seq_list=neg_seq_val, label=0)
    
    X_val = np.concatenate((pos_val_X, neg_val_X), axis=0)
    y_val = np.concatenate((pos_val_y,neg_val_y), axis=0)
    return X_train, y_train, X_test, y_test, X_val, y_val


print ('Building model...')

def model_auto(X_train, y_train, X_val, y_val):
    model = Sequential()
    length = length_seq(X_train)
    print("seq length = ", length)
    model.add(Convolution1D(filters={{choice([50, 100, 150, 200, 250, 300])}},
                        kernel_size={{choice([1, 3, 7, 10, 5, 15, 21, 17])}},
                        input_shape=(length, 4),
                        padding = 'valid'
		        #activation = {{choice(['sigmoid', 'relu'])}})
			))
    
    model.add(MaxPooling1D(pool_size={{choice([0,1,2, 3, 4, 5, 6, 8, 10, 12])}}))
    #model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM(units=256,input_shape = (length, 4),
               return_sequences=True))
    #model.add(Dropout({{uniform(0,1)}}))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    #model.add(LSTM(units=256, 
    #           return_sequences=True))
    #model.add(Dropout({{uniform(0, 1)}}))

    print ('Compiling model...')
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
              loss='binary_crossentropy',
              metrics=['accuracy'])


    model.fit(X_train, y_train, batch_size=16,epochs=10,shuffle = True, verbose=1)
    score, acc = model.evaluate(X_val, y_val, verbose = 0)
    print ('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

        X_train, y_train, X_test, y_test, X_val, y_val = data() 
        best_run, best_model = optim.minimize(model=model_auto,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
    	
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_val, y_val))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)


        print ('Predicting on test data...')

        best_model.save_weights('model_human/my_model1.hdf5')
	#best_model.save_weights('model_bacillus/my_model1.hdf5')
	#best_model.save_weights('model_ecolli/my_model1.hdf5')

        for i in range(5):
                best_model.load_weights('model_human/my_model1.hdf5')
    		#best_model.load_weights('model_bacillus/my_model1.hdf5')
    		#best_model.load_weights('model_ecolli/my_model1.hdf5')
   		#best_model.load_weights('model_ecolli/my_model'+str(i)+'.hdf5')
                y_pred_round = best_model.predict_classes(X_test, batch_size=16, verbose=2)


	
        print ("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred_round))

        confusion= metrics.confusion_matrix(y_test, y_pred_round)
        TP = confusion[1,1]
        TN = confusion[0,0]
        FP = confusion[0,1]
        FN = confusion[1,0]

	#Classification accuracy
        print("Classification accuracy: ", metrics.accuracy_score(y_test, y_pred_round))

	#Correlation co-efficient
        print("Correlation co-efficient: ", metrics.matthews_corrcoef(y_test, y_pred_round))

	#Sensitivity
        print('Sensitivity: ' + str(TP/ float(TP + FN)))

	#Specificity
        print('Specificity: ' + str(TN/ float(TN + FP)))

	

