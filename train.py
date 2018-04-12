import sys
import numpy as np
import h5py
import scipy.io
# np.random.seed(1337) # for reproducibility

import theano
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

def seq_matrix(seq_list, label):
    tensor = np.zeros((len(seq_list), 203, 8))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A' and (j < 100 or j > 102):
                tensor[i][j] = [1, 0, 0, 0, 0, 0, 0, 0]
            if s == 'T' and (j < 100 or j > 102):
                tensor[i][j] = [0, 1, 0, 0, 0, 0, 0, 0]
            if s == 'C' and (j < 100 or j > 102):
                tensor[i][j] = [0, 0, 1, 0, 0, 0, 0, 0]
            if s == 'G' and (j < 100 or j > 102):
                tensor[i][j] = [0, 0, 0, 1, 0, 0, 0, 0]
            if s == '$':
                tensor[i][j] = [0, 0, 0, 0, 0, 0, 0, 0]
            if s == 'A' and (j >= 100 and j <= 102):
                tensor[i][j] = [0, 0, 0, 0, 1, 0, 0, 0]
            if s == 'T' and (j >= 100 and j <= 102):
                tensor[i][j] = [0, 0, 0, 0, 0, 1, 0, 0]
            if s == 'C' and (j >= 100 and j <= 102):
                tensor[i][j] = [0, 0, 0, 0, 0, 0, 1, 0]
            if s == 'G' and (j >= 100 and j <= 102):
                tensor[i][j] = [0, 0, 0, 0, 0, 0, 0, 1]
            j += 1
    if label == 1:
        y = np.ones((len(seq_list), 1))
    else:
        y = np.zeros((len(seq_list), 1))
    return tensor, y


###### main function ######
codon_tis_prior = np.load('dict_piror_front_Gaotrain.npy')
codon_tis_prior = codon_tis_prior.item()

codon_list = []
for c in codon_tis_prior.keys():
    if codon_tis_prior[c] != 'never' and codon_tis_prior[c] >= 1:
        codon_list.append(c)

print ('Loading positive and negative data...')
pos_seq = np.load('data/pos_seq_test.npy')
neg_seq = np.load('data/neg_seq_test_all_upstream.npy')

print('positive', pos_seq.size)
print('neative', neg_seq.size)
#print(pos_seq)
#np.load(open(r'C:\Final Runs\lineTank.npy', 'rb'))

to_train1 = int(pos_seq.size* 0.8)
to_train2 = int(neg_seq.size* 0.8)
    # tweak to match with batch_size
batch_size =16
to_train1 -= to_train1 % batch_size
to_train2 -= to_train1 % batch_size

pos_seq_train = pos_seq[:to_train1]
neg_seq_train = neg_seq[:to_train2]
pos_seq_test = pos_seq[to_train1:]
neg_seq_test = neg_seq[to_train2:]




pos_codon = []
neg_codon = []

#using divided test data
for s in pos_seq_test:
    if s[100:103] in codon_list:
        pos_codon.append(codon_tis_prior[s[100:103]])
for s in neg_seq_test:
    if s[100:103] in codon_list:
        neg_codon.append(codon_tis_prior[s[100:103]])

pos_codon = np.array(pos_codon)
neg_codon = np.array(neg_codon)
codon = np.concatenate((pos_codon, neg_codon)).reshape((len(pos_codon) + len(neg_codon), 1))



#FOr training data
pos_seq_train1 = []
neg_seq_train1 = []

#using divided train data here
for s in pos_seq_train:
    if s[100:103] in codon_list:
        pos_seq_train1.append(s)
for s in neg_seq_train:
    if s[100:103] in codon_list:
        neg_seq_train1.append(s)

print (str(len(pos_seq_train1)) + ' positive train data loaded...')
print (str(len(neg_seq_train1)) + ' negative train data loaded...')


pos_train_X, pos_train_y = seq_matrix(seq_list=pos_seq_train1, label=1)
neg_train_X, neg_train_y = seq_matrix(seq_list=neg_seq_train1, label=0)
X_train = np.concatenate((pos_train_X, neg_train_X), axis=0)
y_train = np.concatenate((pos_train_y, neg_train_y), axis=0)




#For test data
pos_seq_test1 = []
neg_seq_test1 = []

#using divided test data here
for s in pos_seq_test:
    if s[100:103] in codon_list:
        pos_seq_test1.append(s)
for s in neg_seq_test:
    if s[100:103] in codon_list:
        neg_seq_test1.append(s)

print (str(len(pos_seq_test1)) + ' positive test data loaded...')
print (str(len(neg_seq_test1)) + ' negative test data loaded...')

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test1, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test1, label=0)
X_test = np.concatenate((pos_test_X, neg_test_X), axis=0)
y_test = np.concatenate((pos_test_y,neg_test_y), axis=0)

print ('Building model...')

model = Sequential()
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=8,
                        input_length=203,
                        border_mode='valid',
                        W_constraint=maxnorm(3),
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=3))
model.add(Dropout(p=0.21370950078747658))
model.add(LSTM(output_dim=256,
               return_sequences=True))
model.add(Dropout(p=0.7238091317104384))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

print ('Compiling model...')
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print ('Predicting on test data...')
y_test_pred_n = np.zeros((len(y_test), 1))
y_test_pred_p = np.zeros((len(y_test), 1))


model.save_weights('ALS_model/my_model_10.hdf5')



for i in range(10):
   # model.load_weights('ALS_model/my_model.hdf5')
    model.load_weights('ALS_model/my_model_'+str(i)+'.hdf5')
    y_test_pred = model.predict(X_test, verbose=1)
    y_test_pred_n += y_test_pred
    y_test_pred_p += y_test_pred * codon

y_test_pred_n = y_test_pred_n / 32
y_test_pred_p = y_test_pred_p / 32



#print ('Accuracy Score: ' +str(accuracy_score(y_test, y_test_pred.round())))
#print ('Confusion Matrix: ' +str(confusion_matrix(y_test, y_test_pred.round())))

#print(classification_report(y_test, y_test_pred.round()))


model.fit(X_train, y_train, batch_size=16,nb_epoch=10)
score = model.evaluate(X_test, y_test, batch_size=16)
print score


print ('Perf without prior, AUC: ' + str(roc_auc_score(y_test, y_test_pred_n)))
print ('Perf without prior, AUPR: ' + str(average_precision_score(y_test, y_test_pred_n)))
print ('Perf with prior, AUC: ' + str(roc_auc_score(y_test, y_test_pred_p)))
print ('Perf with prior, AUPR: ' + str(average_precision_score(y_test, y_test_pred_p)))


#confusion matrix
print metrics.confusion_matrix(y_test, y_test_pred.round())

#Save confusion matrix and slice into four pieces
confusion= metrics.confusion_matrix(y_test, y_test_pred.round())
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

#Classification accuracy
print ('Classification accuracy: ' + str((TP + TN)/ float(TP + TN + FP + FN)))
print(metrics.accuracy_score(y_test, y_test_pred.round()))

#Sensitivity
print('Sensitivity: ' + str(TP/ float(TP + TN)))

#Specificity
print('Specificity: ' + str(TN/ float(TN + FP)))














