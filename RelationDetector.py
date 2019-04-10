#!pip install git+https://www.github.com/keras-team/keras-contrib.git

from __future__ import print_function
from keras.preprocessing import sequence
from keras_self_attention import SeqSelfAttention
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, TimeDistributed, Bidirectional, Flatten
from keras.datasets import imdb
from keras.models import load_model
import random
from keras_contrib.layers import CRF

import numpy as np
import os
import File_Parser as fp
from keras import optimizers

from keras.models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 50
#batch_size = 32

OneEHR, OneEHRLabels, OneEHRRel = fp.getCompleteWordFeature()

train_data = np.array(OneEHRLabels)
gold_label = np.array(OneEHRRel)
print("gold label",gold_label[0])
print("Shape of train data",train_data.shape)
print("Shape of gold label",gold_label.shape)
print("Length of train data : ",len(train_data))

#Split into training and testing sets
x_train=train_data[:500]
x_test=train_data[500:]

label_data=[]
tempOpS=[]
tempOpW=[]

gold_label = np.array(gold_label)
label_data=gold_label
y_train=label_data[:500]
y_test=label_data[500:]

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

train_data=np.array(train_data)

print("X train : ",x_train.shape)
print("X test : ",x_test.shape)
print ("Y train : ",y_train.shape)
print("Y test : ",y_test.shape)

print('Pad sequences (samples x time)')

#Add padding to make the size of each EHR equal
x_train = sequence.pad_sequences(x_train, padding='post',maxlen=maxlen, value=0)
x_test = sequence.pad_sequences(x_test,padding='post', maxlen=maxlen, value=0)
y_test = sequence.pad_sequences(y_test,padding='post', maxlen=maxlen, value=0)

train_data=sequence.pad_sequences(train_data,padding='post', maxlen=maxlen, value=0)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Build model...')

#Initialize model
model = Sequential()

#Use Bi directional Long Short Term Memory model
model.add(Bidirectional(LSTM(20,return_sequences=True),merge_mode = 'concat'))

#Use attension layer
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(units=9))

model.add(Dense(9, activation='softmax'))
sgd = optimizers.SGD(lr=10, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(np.array(x_train), np.array(y_train),
          epochs=50
          #validation_data=(x_test, y_test)
          )

print('Summary : ',model.summary())

#Save model for future use
model.save('relationModel_e1000.h5')

#Test results against testing data
y_pred = model.predict(x_train)
print("Y Pred :", y_pred.shape)
print("Y ", y_pred)

