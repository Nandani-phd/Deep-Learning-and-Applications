#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle as p
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LSTM, Input, Masking,Dense


# In[2]:


file_path = "group1/CV/"
classes = ['ba', 'hai','hI', 'ka','ni']
data_split = ['Test', 'Train']


# In[3]:


Y_train = []
X_train = []
X_test = []
Y_test = []

dir_list = os.listdir(file_path)
print(dir_list)
for j in data_split:
    for i in classes:
        for filename in glob.glob(os.path.join(file_path + '/' + i + '/' + j, '*.mfcc')):
            with open(filename, 'r') as f:
                if (j == "Train"):
                    # print(filename)
                    f = open(filename)
                    parsed = f.readlines()
                    parsed = [np.array([float(val) for val in line.split(" ") if val!="\n"]) for line in parsed]
                    X_train.append(np.array(parsed))
                    Y_train.append(str(i))
                if (j == 'Test'):
                    # print(filename)
                    f = open(filename)
                    parsed = f.readlines()
                    parsed = [np.array([float(val) for val in line.split(" ") if val!="\n"]) for line in parsed]
                    X_test.append(np.array(parsed))
                    Y_test.append(str(i))
print("Training Set has {} samples each with dimension {} but variable length".format(len(X_train),X_train[0][0].shape))
print("Test Set has {} samples each with dimension {} but variable length".format(len(X_test),X_test[0][0].shape))


# padding input
max_len = max([len(val) for val in X_train])
print("Max_seq length: ",max_len)
def padd_it(data):
    for i in range(len(data)):
        if len(data[i]) <= max_len:
            seq = np.repeat(np.array([-1]*39).reshape(-1,39),int(max_len-len(data[i])),0)
            data[i] = np.append(data[i],seq,axis=0)
    return data
# data preprocessing steps to prepare the input data for training and testing a model
print("Before Padding shape of first element in train is {}".format((X_train[0]).shape))
X_test = np.asarray(padd_it(X_test))
X_train = np.asarray(padd_it(X_train))
print("After Padding shape of first element in train is {}".format((X_train[0]).shape))


# In[4]:


type(X_train)


# In[5]:


X_train.shape


# In[6]:


Y_train=np.array(Y_train).reshape(-1, 1)


# In[7]:


Y_test=np.array(Y_test).reshape(-1, 1)


# In[8]:


from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
#define ordinal encoding
encoder = OrdinalEncoder()
# transform data
yt= encoder.fit_transform(Y_train)
print(yt)


# In[9]:


ytest= encoder.fit_transform(Y_test)
print(ytest)


# In[ ]:





# In[10]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# In[11]:


es= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[12]:


mask_value=-1
batch_size=128

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model.add(LSTM(units=64))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()


# In[13]:


history=model.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[14]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc=model.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(trainAcc)

print('\nEvaluation of model on test data:')
testAcc=model.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(testAcc)

print('\nPredictions for test data:')
testProb=model.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred=np.argmax(testProb,axis=1)

confusionMatrix=tf.math.confusion_matrix(ytest,pred)
print(confusionMatrix)


# In[15]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[16]:


es1= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[17]:


model1 = Sequential()
model1.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model1.add(LSTM(units=64,dropout=0.2))
model1.add(Dense(units=5, activation='softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[18]:


model1.summary()


# In[19]:


history1=model1.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[38]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc1=model1.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(trainAcc1)

print('\nEvaluation of model on test data:')
testAcc1=model1.evaluate(X_test, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(testAcc1)

print('\nPredictions for test data:')
testProb1=model1.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred1=np.argmax(testProb1,axis=1)


#3-3
confusionMatrix1=tf.math.confusion_matrix(ytest,pred1)
print(confusionMatrix1)


# In[20]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc1=model1.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(trainAcc1)

print('\nEvaluation of model on test data:')
testAcc1=model1.evaluate(X_test, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(testAcc1)

print('\nPredictions for test data:')
testProb1=model1.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred1=np.argmax(testProb1,axis=1)


#3-3
confusionMatrix1=tf.math.confusion_matrix(ytest,pred1)
print(confusionMatrix1)


# In[21]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[22]:


es2= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[23]:


model2 = Sequential()
model2.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model2.add(LSTM(units=32))
model2.add(Dense(units=5, activation='softmax'))
model2.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[24]:


history2=model2.fit(X_train, yt, epochs=1000,callbacks=[es2], batch_size=batch_size)


# In[25]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc2=model2.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model2.metrics_names)
print(trainAcc2)

print('\nEvaluation of model on test data:')
testAcc2=model2.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model2.metrics_names)
print(testAcc2)

print('\nPredictions for test data:')
testProb2=model2.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred2=np.argmax(testProb2,axis=1)


#3-3
confusionMatrix2=tf.math.confusion_matrix(ytest,pred2)
print(confusionMatrix2)


# In[26]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[27]:


model3 = Sequential()
model3.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model3.add(LSTM(units=128))
model3.add(Dense(units=5, activation='softmax'))
model3.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[28]:


model3.summary()


# In[29]:


history31=model3.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[37]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc3=model3.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(trainAcc3)

print('\nEvaluation of model on test data:')
testAcc3=model3.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(testAcc3)

print('\nPredictions for test data:')
testProb3=model3.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred3=np.argmax(testProb3,axis=1)


#3-3
confusionMatrix3=tf.math.confusion_matrix(ytest,pred3)
print(confusionMatrix3)


# In[30]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc3=model3.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(trainAcc3)

print('\nEvaluation of model on test data:')
testAcc3=model3.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(testAcc3)

print('\nPredictions for test data:')
testProb3=model3.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred3=np.argmax(testProb3,axis=1)


#3-3
confusionMatrix3=tf.math.confusion_matrix(ytest,pred3)
print(confusionMatrix3)


# In[31]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history31.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[32]:


model41 = Sequential()
model41.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model41.add(LSTM(units=128,dropout=0.2))
model41.add(Dense(units=5, activation='softmax'))
model41.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[33]:


model41.summary()


# In[34]:


history41=model41.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[35]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc41=model41.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model41.metrics_names)
print(trainAcc41)

print('\nEvaluation of model on test data:')
testAcc41=model41.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model41.metrics_names)
print(testAcc41)

print('\nPredictions for test data:')
testProb41=model41.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred41=np.argmax(testProb41,axis=1)


#3-3
confusionMatrix41=tf.math.confusion_matrix(ytest,pred41)
print(confusionMatrix41)


# In[36]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history41.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

