import sys
import numpy as np
import h5py
import scipy.io
# np.random.seed(1337) # for reproducibility
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
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
import Bio

def seq_matrix(seq_list, label):
    tensor = np.zeros((len(seq_list), 251, 4))  #replace 81 with 251 for human data
    for i in range(len(seq_list)):
        seq = seq_list[i]
        # print seq

        j = 0
        for s in seq:
            if s == 'A':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'C':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'G':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'T':
                tensor[i][j] = [0, 0, 0, 1]

            j += 1
    if label == 1:
        y = np.ones((len(seq_list), 1))
    else:
        y = np.zeros((len(seq_list), 1))

    return tensor, y



#Reading fasta files
def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))


print('Loading Positive and negative data...')

# with open('data/bacillus.fa') as fp:
with open('data/ecoli.fa') as fp:
#with open('data/human_nontata.fa') as fp:
    pos_seq = []
    for name, seq in read_fasta(fp):
        pos_seq.append(seq)
print('pos_seq: ', len(pos_seq))


# with open('data/bacillus_non.fa') as fp:
with open('data/ecoli_non.fa') as fp:

#with open('data/human_nontata_non.fa') as fp:
    neg_seq = []
    for name, seq in read_fasta(fp):
        neg_seq.append(seq)

print('neg_seq: ', len(neg_seq))

to_train1 = int(len(pos_seq)* 0.8)
to_train2 = int(len(neg_seq)* 0.8)
    # tweak to match with batch_size
batch_size =16
to_train1 -= to_train1 % batch_size
to_train2 -= to_train1 % batch_size

pos_seq_train = pos_seq[:to_train1]
neg_seq_train = neg_seq[:to_train2]
pos_seq_test = pos_seq[to_train1:]
neg_seq_test = neg_seq[to_train2:]

print (str(len(pos_seq_train)) + ' positive train data loaded...')
print (str(len(neg_seq_train)) + ' negative train data loaded...')

pos_train_X, pos_train_y = seq_matrix(seq_list=pos_seq_train, label=1)
#print("pos_train_X: ", pos_train_X)
#print("pos_train_y: ", pos_train_y)
neg_train_X, neg_train_y = seq_matrix(seq_list=neg_seq_train, label=0)
#print("neg_train_X: ", neg_train_X)
X_train = np.concatenate((pos_train_X, neg_train_X), axis=0)
y_train = np.concatenate((pos_train_y, neg_train_y), axis=0)

print (str(len(pos_seq_test)) + ' positive test data loaded...')
print (str(len(neg_seq_test)) + ' negative test data loaded...')

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test, label=0)
#print("pos_test_X: ", pos_test_X)
#print("pos_test_y: ", pos_test_y)
X_test = np.concatenate((pos_test_X, neg_test_X), axis=0)
y_test = np.concatenate((pos_test_y,neg_test_y), axis=0)

print ('Building model...')

model = Sequential()
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=4,
                        input_length=251,
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

#model.save_weights('model_human/my_model.hdf5')
# model.save_weights('model_bacillus/my_model.hdf5')
model.save_weights('model_ecolli/my_model.hdf5')

for i in range(10):
    #model.load_weights('model_human/my_model.hdf5')
    # model.load_weights('model_bacillus/my_model.hdf5')
    model.load_weights('model_ecolli/my_model.hdf5')
   #  model.load_weights('model/my_model_'+str(i)+'.hdf5')
    y_test_pred = model.predict(X_test, verbose=1)

model.fit(X_train, y_train, batch_size=16,nb_epoch=10)
score = model.evaluate(X_test, y_test, batch_size=16)
print score

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










