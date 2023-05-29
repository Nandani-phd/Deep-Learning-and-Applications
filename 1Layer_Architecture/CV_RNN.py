#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle as p
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Input, Masking


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


file_path = "group1/CV/"
classes = ['ba', 'hai','hI', 'ka','ni']
data_split = ['Test', 'Train']


# In[6]:


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


# In[ ]:


type(X_train)


# In[ ]:


X_train.shape


# In[7]:


Y_train=np.array(Y_train).reshape(-1, 1)


# In[8]:


Y_test=np.array(Y_test).reshape(-1, 1)


# In[9]:


from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
#define ordinal encoding
encoder = OrdinalEncoder()
# transform data
yt= encoder.fit_transform(Y_train)
print(yt)


# In[10]:


ytest= encoder.fit_transform(Y_test)
print(ytest)


# In[11]:


CV_RNN_Model1=Sequential()
CV_RNN_Model1.add(Masking(mask_value=-1,input_shape=X_train.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
CV_RNN_Model1.add(SimpleRNN(32, return_sequences=False))
CV_RNN_Model1.add(Dropout(0.2))
CV_RNN_Model1.add(Dense(5, activation='softmax'))
CV_RNN_Model1.summary()


# In[12]:


CV_RNN_Model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=3)
history = CV_RNN_Model1.fit(X_train, yt, batch_size=32,callbacks=[callback] ,epochs=500,verbose=1)



# In[ ]:


CV_RNN_Model1.save('/content/drive/MyDrive/CV_RNN_Model1.h5')


# In[ ]:


#Save Model and Model Structure
CV_RNN_Model1.save('/content/drive/MyDrive/DLA6/CV_SAVE_Model/RNN_CV1L32.h5')
f=open('/content/drive/MyDrive/DLA6/CV_SAVE_Model/RNN_CV_HIST1L32.pckl','wb')
p.dump(history.history,f)
f.close()


# In[ ]:


# Extract the loss values from the history object
training_loss = history.history['loss']


# In[ ]:


#Plot the training loss against epochs
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, 'b', label='Average Training Error')
plt.xlabel('Epochs')
plt.ylabel('Average Training Error')
plt.title('Average Training Error vs. Epochs')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


CV_RNN_Model1.evaluate(X_train,yt)
CV_RNN_Model1.evaluate(X_test,ytest)


# In[ ]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc1=CV_RNN_Model1.evaluate(x=X_train,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(CV_RNN_Model1.metrics_names)
print(trainAcc1)
print('\nEvaluation of model on test data:')
testAcc1=CV_RNN_Model1.evaluate(x=X_test, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(CV_RNN_Model1.metrics_names)
print(testAcc1)
print('\nPredictions for test data:')
testProb1=CV_RNN_Model1.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred1=np.argmax(testProb1,axis=1)
confusionMatrix1=tf.math.confusion_matrix(ytest,pred1)
print(confusionMatrix1)


# In[ ]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[13]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# In[14]:


es= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[19]:


mask_value=-1
batch_size=128

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model.add(SimpleRNN(units=64))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()


# In[20]:


history=model.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[23]:


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


# In[24]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[26]:


es1= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[31]:


model1 = Sequential()
model1.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model1.add(SimpleRNN(units=64,dropout=0.2))
model1.add(Dense(units=5, activation='softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[32]:


model1.summary()


# In[33]:


history1=model1.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[35]:


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


# In[36]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[40]:


es2= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[39]:


model2 = Sequential()
model2.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model2.add(SimpleRNN(units=32))
model2.add(Dense(units=5, activation='softmax'))
model2.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[41]:


history2=model2.fit(X_train, yt, epochs=1000,callbacks=[es2], batch_size=batch_size)


# In[42]:


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


# In[43]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[58]:


es21= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[59]:


model21 = Sequential()
model21.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model21.add(SimpleRNN(units=32))
model21.add(Dense(units=5, activation='softmax'))
model21.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[60]:


history21=model21.fit(X_train, yt, epochs=1000,callbacks=[es2], batch_size=batch_size)


# In[61]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc21=model21.evaluate(X_train, yt,batch_size=1, verbose="auto",callbacks=None)
print(model21.metrics_names)
print(trainAcc21)

print('\nEvaluation of model on test data:')
testAcc21=model21.evaluate(X_test, ytest, batch_size=1, verbose="auto",callbacks=None)
print(model21.metrics_names)
print(testAcc21)

print('\nPredictions for test data:')
testProb21=model21.predict(X_test, batch_size=1, verbose="auto", callbacks=None)
pred21=np.argmax(testProb21,axis=1)


#3-3
confusionMatrix21=tf.math.confusion_matrix(ytest,pred21)
print(confusionMatrix21)


# In[62]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history21.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[63]:


es2= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[64]:


model2 = Sequential()
model2.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model2.add(SimpleRNN(units=32,dropout=0.2))
model2.add(Dense(units=5, activation='softmax'))
model2.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[65]:


history2=model2.fit(X_train, yt, epochs=1000,callbacks=[es2], batch_size=batch_size)


# In[66]:


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


# In[67]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:





# In[47]:


model3 = Sequential()
model3.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model3.add(SimpleRNN(units=128))
model3.add(Dense(units=5, activation='softmax'))
model3.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[48]:


model3.summary()


# In[49]:


history31=model3.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[50]:


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


# In[51]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history31.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[52]:


model41 = Sequential()
model41.add(Masking(mask_value=mask_value, input_shape=X_train.shape[1:]))
model41.add(SimpleRNN(units=128,dropout=0.2))
model41.add(Dense(units=5, activation='softmax'))
model41.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[53]:


model41.summary()


# In[54]:


history41=model41.fit(X_train, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[56]:


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


# In[57]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history41.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

