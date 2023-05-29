#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from tensorflow.keras.layers import Embedding, LSTM,Activation, Dense, Dropout, SimpleRNN, Masking, Input, Activation
from tensorflow.keras.activations import softmax


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Embedding, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from keras import initializers
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.models as m


# In[3]:


path = "group1/Handwritten/"
#os.listdir("group1/Handwritten/ai/train")


# In[4]:


def read_data(path):
    tr_path = path+'/train/'
    test_path = path+'/dev/'
    train = pd.concat([pd.read_csv(tr_path+i, sep=" ", header=None) for i in os.listdir(tr_path)])
    test = pd.concat([pd.read_csv(test_path+i, sep=" ", header=None) for i in os.listdir(test_path)])
    return train, test


# In[5]:


tr_ai, test_ai= read_data(path+'/ai')
tr_bA, test_bA= read_data(path+'/bA')
tr_chA, test_chA= read_data(path+'/chA')
tr_dA, test_dA= read_data(path+'/dA')
tr_tA, test_tA= read_data(path+'/tA')


# In[6]:


tr_chA.drop(0, inplace=True, axis=1)
tr_dA.drop(0, inplace=True, axis=1)
tr_ai.drop(0, inplace=True, axis=1)
tr_bA.drop(0, inplace=True, axis=1)
tr_tA.drop(0, inplace=True, axis=1)


# In[7]:


test_chA.drop(0, inplace=True, axis=1)
test_dA.drop(0, inplace=True, axis=1)
test_ai.drop(0, inplace=True, axis=1)
test_bA.drop(0, inplace=True, axis=1)
test_tA.drop(0, inplace=True, axis=1)


# In[8]:


tr_ai=tr_ai.dropna(axis=1,how='all')
test_ai=test_ai.dropna(axis=1,how='all')
tr_bA=tr_bA.dropna(axis=1,how='all')
test_bA=test_bA.dropna(axis=1,how='all')
tr_chA=tr_chA.dropna(axis=1,how='all')
test_chA=test_chA.dropna(axis=1,how='all')
tr_dA=tr_dA.dropna(axis=1,how='all')
test_dA=test_dA.dropna(axis=1,how='all')
tr_tA=tr_tA.dropna(axis=1,how='all')
test_tA=test_tA.dropna(axis=1,how='all')


# In[9]:


#import pandas as pd

tr_data = pd.concat([tr_chA, tr_tA, tr_ai, tr_bA, tr_dA])
test_data = pd.concat([test_chA, test_tA, test_ai, test_bA, test_dA])
# save to training.csv file
#tr_data.to_csv('training.csv', index=False)


# In[137]:


nan_list=[np.nan]*test_data.shape[0]
#nan_list


# In[11]:


for i in range(test_data.shape[1],tr_data.shape[1]+1):
    test_data[i]=nan_list


# In[12]:


tr_data=tr_data.replace(np.nan,0)
test_data=test_data.replace(np.nan,0)


# In[13]:


df =tr_data
# Create an empty list to store the x, y point pairs
tr = []


# Loop over each row in the DataFrame
for index, row in df.iterrows():
    # Create a list to store the x, y pairs for this row
    point_row = []
    
    
    # Loop over each value in the row, skipping the first column
    for i in range(1, len(row), 2):
        # Get the x, y pair for this point
        x = row[i]
        y = row[i+1]
        l=[]
        #print(type(x))
        l.append(x)
        l.append(y)
        l=np.array(l,dtype=np.float32)
        # Add the x, y pair to the list for this row
        point_row.append(l)
    
    # Add the list of x, y pairs for this row to the points list
    tr.append(point_row)

# Create a new DataFrame from the points list
train = pd.DataFrame(tr)

# Save the new DataFrame to a CSV file
#df_points.to_csv('testHw.csv', index=False, header=False)


# In[14]:


df =test_data
# Create an empty list to store the x, y point pairs
tst = []


# Loop over each row in the DataFrame
for index, row in df.iterrows():
    # Create a list to store the x, y pairs for this row
    point_row = []
    
    
    # Loop over each value in the row, skipping the first column
    for i in range(1, len(row), 2):
        # Get the x, y pair for this point
        x = row[i]
        y = row[i+1]
        l=[]
        #print(type(x))
        l.append(x)
        l.append(y)
        l=np.array(l,dtype=np.float32)
        # Add the x, y pair to the list for this row
        point_row.append(l)
    
    # Add the list of x, y pairs for this row to the points list
    tst.append(point_row)

# Create a new DataFrame from the points list
test = pd.DataFrame(tst)

# Save the new DataFrame to a CSV file
#df_points.to_csv('testHw.csv', index=False, header=False)


# In[15]:


trainLabel=pd.read_csv('label1.csv')


# In[16]:


trainLabel.drop('label',inplace=True,axis=1)


# In[134]:


yt=trainLabel.values
#yt
#print(len(yt[0]))


# In[18]:


testLabel=pd.read_csv('Test_label1.csv')


# In[19]:


testLabel.drop('label',inplace=True,axis=1)


# In[136]:


ytest=testLabel.values
#ytest
#print(len(ytest))


# In[22]:


tensortrain=tf.convert_to_tensor(tr)


# In[23]:


tensortest=tf.convert_to_tensor(tst)


# #Model 1- es-35, no dropout,units-64

# In[26]:


es= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[25]:


mask_value=0.0
batch_size=128

model = Sequential()
model.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model.add(LSTM(units=64))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()


# In[27]:


history=model.fit(tensortrain, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[28]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc=model.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(trainAcc)

print('\nEvaluation of model on test data:')
testAcc=model.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(testAcc)

print('\nPredictions for test data:')
testProb=model.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred=np.argmax(testProb,axis=1)

confusionMatrix=tf.math.confusion_matrix(ytest,pred)
print(confusionMatrix)


# In[57]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#Model 1- es-35, dropout-0.2,units-64


# In[36]:


es1= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[37]:


model1 = Sequential()
model1.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model1.add(LSTM(units=64,dropout=0.2))
model1.add(Dense(units=5, activation='softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[38]:


model1.summary()


# In[39]:


history1=model1.fit(tensortrain, yt, epochs=1000,callbacks=[es1], batch_size=batch_size)


# In[40]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc1=model1.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(trainAcc1)

print('\nEvaluation of model on test data:')
testAcc1=model1.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model1.metrics_names)
print(testAcc1)

print('\nPredictions for test data:')
testProb1=model1.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred1=np.argmax(testProb1,axis=1)


#3-3
confusionMatrix1=tf.math.confusion_matrix(ytest,pred1)
print(confusionMatrix1)


# In[58]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[42]:


es2= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[43]:


model2 = Sequential()
model2.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model2.add(LSTM(units=32))
model2.add(Dense(units=5, activation='softmax'))
model2.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[44]:


history2=model2.fit(tensortrain, yt, epochs=1000,callbacks=[es2], batch_size=batch_size)


# In[45]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc2=model2.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model2.metrics_names)
print(trainAcc2)

print('\nEvaluation of model on test data:')
testAcc2=model2.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model2.metrics_names)
print(testAcc2)

print('\nPredictions for test data:')
testProb2=model2.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred2=np.argmax(testProb2,axis=1)


#3-3
confusionMatrix2=tf.math.confusion_matrix(ytest,pred2)
print(confusionMatrix2)


# In[59]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[46]:


es3= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[47]:


model3 = Sequential()
model3.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model3.add(LSTM(units=128))
model3.add(Dense(units=5, activation='softmax'))
model3.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[48]:


model3.summary()


# In[49]:


history31=model3.fit(tensortrain, yt, epochs=1000,callbacks=[es3], batch_size=batch_size)


# In[50]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc3=model3.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(trainAcc3)

print('\nEvaluation of model on test data:')
testAcc3=model3.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model3.metrics_names)
print(testAcc3)

print('\nPredictions for test data:')
testProb3=model3.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred3=np.argmax(testProb3,axis=1)


#3-3
confusionMatrix3=tf.math.confusion_matrix(ytest,pred3)
print(confusionMatrix3)


# In[61]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history31.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[54]:


es4= EarlyStopping(monitor='loss',min_delta=0.0001, patience=35,verbose=1)


# In[51]:


model4 = Sequential()
model4.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model4.add(LSTM(units=128,dropout=0.2))
model4.add(Dense(units=5, activation='softmax'))
model4.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[ ]:


model41 = Sequential()
model41.add(Masking(mask_value=mask_value, input_shape=(161, 2)))
model41.add(SimpleRNN(units=128,dropout=0.2))
model41.add(Dense(units=5, activation='softmax'))
model41.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# In[53]:


model4.summary()


# In[ ]:


history41=model41.fit(tensortrain, yt, epochs=1000,callbacks=[es], batch_size=batch_size)


# In[ ]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc41=model41.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model41.metrics_names)
print(trainAcc41)

print('\nEvaluation of model on test data:')
testAcc41=model41.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model41.metrics_names)
print(testAcc41)

print('\nPredictions for test data:')
testProb41=model41.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred41=np.argmax(testProb41,axis=1)


#3-3
confusionMatrix41=tf.math.confusion_matrix(ytest,pred41)
print(confusionMatrix41)


# In[55]:


history4=model4.fit(tensortrain, yt, epochs=1000,callbacks=[es4], batch_size=batch_size)


# In[56]:


#Evaluating the model
print('\nEvaluation of model on train data:')
trainAcc4=model4.evaluate(x=tensortrain,y=yt,batch_size=1, verbose="auto",callbacks=None)
print(model4.metrics_names)
print(trainAcc4)

print('\nEvaluation of model on test data:')
testAcc4=model4.evaluate(x=tensortest, y=ytest, batch_size=1, verbose="auto",callbacks=None)
print(model4.metrics_names)
print(testAcc4)

print('\nPredictions for test data:')
testProb4=model4.predict(tensortest, batch_size=1, verbose="auto", callbacks=None)
pred4=np.argmax(testProb4,axis=1)


#3-3
confusionMatrix4=tf.math.confusion_matrix(ytest,pred4)
print(confusionMatrix4)


# In[62]:


plt.figure(figsize=(8, 6))
#plt.plot(history8.history['accuracy'])
plt.plot(history4.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




