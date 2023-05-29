import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os,sys
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,Flatten,Reshape
from tensorflow.keras.models import Model
from keras import initializers
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle as p
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix 

#Arch1-256
#Arch2-128
#Arch3-64
#Arch4-32

#All layers have 1024,512,128

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=100)
#print('Initializer: ',initializer)

def Upload_Dataset(Dataset_Path):
    
    Path_train_Dataset = Dataset_Path+"/train"
    Path_test_Dataset = Dataset_Path+"/test"
    Path_val_Dataset = Dataset_Path+"/val"
    
    Input_train_Dataset, Input_test_Dataset, Input_val_Dataset = [], [], []
    Output_train, Output_test, Output_val = [], [], []

    for j in os.listdir(Path_train_Dataset):

        if j != ".DS_Store":
            for k in os.listdir(Path_train_Dataset+"/"+j):
                Input_train_Dataset.append(cv2.imread(Path_train_Dataset+"/"+j+"/"+k, cv2.IMREAD_GRAYSCALE))
                if(j == '6'):
                    Output_train.append(int(2))
                elif(j == '7'):
                    Output_train.append(int(3))
                elif(j == '9'):
                    Output_train.append(int(4))
                else:
                    Output_train.append(int(j))

            for k in os.listdir(Path_test_Dataset+"/"+j):
                Input_test_Dataset.append(cv2.imread(Path_test_Dataset+"/"+j+"/"+k, cv2.IMREAD_GRAYSCALE))
                if(j=='6'):
                    Output_test.append(int(2))
                elif(j=='7'):
                    Output_test.append(int(3))
                elif(j=='9'):
                    Output_test.append(int(4))
                else:
                    Output_test.append(int(j))

            for k in os.listdir(Path_val_Dataset+"/"+j):
                Input_val_Dataset.append(cv2.imread(Path_val_Dataset+"/"+j+"/"+k, cv2.IMREAD_GRAYSCALE))
                if(j=='6'):
                    Output_val.append(int(2))
                elif(j=='7'):
                    Output_val.append(int(3))
                elif(j=='9'):
                    Output_val.append(int(4))
                else:
                    Output_val.append(int(j))
        
    #print(Output_train)            
    Input_train_Dataset, Input_test_Dataset, Input_val_Dataset = np.array(Input_train_Dataset), np.array(Input_test_Dataset), np.array(Input_val_Dataset)
    #Output_train, Output_test, Output_val = np.array(list(map(int, Output_train))), np.array(list(map(int, Output_test))), np.array(list(map(int, Output_val)))
    Output_train, Output_test, Output_val = np.array(Output_train), np.array(Output_test), np.array(Output_val)
    #print(Output_train) 
    return Input_train_Dataset, Input_test_Dataset,Input_val_Dataset, Output_train, Output_test, Output_val



#providing path of  MNIST dataset
Dataset_Path = "/Users/vds/Downloads/Group_1"


#calling Upload_Dataset function
Input_train_Dataset, Input_test_Dataset,Input_val_Dataset, Output_train, Output_test, Output_val= Upload_Dataset(Dataset_Path)
#print(Input_train_Dataset.shape,'\n')
xTrain=np.reshape(Input_train_Dataset,(Input_train_Dataset.shape[0],784))
xVal=np.reshape(Input_val_Dataset,(Input_val_Dataset.shape[0],784))
encodingDim=32
print(Input_train_Dataset.shape)

inp=Input(shape=(784,))

encodedOut=Dense(encodingDin,activation='sigmoid')(inp)

decodedOut=Dense(784,activation='sigmoid')(encodedOut)

a1Encoder=Model(innp,encodedOut,name='encoder')

a1Autoencoder=Model(xInp,out,name='AutoencoderA1')
a1Autoencoder.summary()

es = EarlyStopping(monitor='loss',min_delta=0.0001, patience=3,verbose=1)

a1Autoencoder.compile(optimizer, loss='mse', metrics=['accuracy'])
ae1history=a1Autoencoder.fit(x = xTrain, y = xTrain, batch_size = 32, epochs = 100, verbose = "auto", callbacks=[es], validation_data=(Input_val_Dataset,  Output_val), validation_batch_size=1)



#xTrain=np.reshape(Input_train_Dataset,(Input_train_Dataset.shape[0],784))
#xInp=Input(input_shape=(28,28),name="img")
#inp=Flatten()(xInp)
xInp=Input(shape=(28,28),name="img")
inp=Flatten(input_shape=(28, 28), name='InputLayer')(xInp)
#xInp=Input(input_shape=(28,28),name="img")
#inp=Input(shape=(784,))(xInp)


#xInp=Flatten(input_shape=(28,28),name='input')
#inp=Input(shape=(784,))(xInp)
encodedOut=Dense(encodingDim,activation='sigmoid')(inp)

a1Encoder=Model(xInp,encodedOut,name='encoder')

decodedOut=Dense(784,activation='sigmoid')(encodedOut)
out=Reshape((28,28))(decodedOut)
optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,name='Adam')

a1Autoencoder=Model(xInp,out,name='AutoencoderA1')
a1Autoencoder.summary()

es = EarlyStopping(monitor='loss',min_delta=0.0001, patience=10,verbose=1)

a1Autoencoder.compile(optimizer, loss='mse', metrics=['accuracy'])
ae1history=a1Autoencoder.fit(x = Input_train_Dataset, y = Input_train_Dataset, batch_size = 32, epochs = 10000, verbose = "auto", callbacks=[es], validation_data=(Input_val_Dataset,  Output_val), validation_batch_size=1)


trainEncode=a1Encoder.predict(Input_train_Dataset)
trainReconstruct=a1Autoencoder.predict(Input_train_Dataset)
'''
model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28), name='InputLayer'),
        layers.Dense(1024, activation="sigmoid", name="Hlayer1",kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(512, activation="sigmoid", name="Hlayer2",kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(256, activation="sigmoid", name="Hlayer3",kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(5, activation="softmax", name="output"),
        ])
model.summary()

adam =Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,name='Adam')


model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='loss',min_delta=0.0001, patience=10,verbose=1)

model_fit = model.fit(x = pcaTrainrecons4, y = Output_train, batch_size = 32, epochs = 10000, verbose = "auto", callbacks=[es], validation_data=(Input_val_Dataset,  Output_val), validation_batch_size=1)
model.save('/Users/vds/Downloads/modelAssign4/ae256.h5')
f=open('/Users/vds/Downloads/modelAssign4/Histae256.pckl','wb')
p.dump(model_fit.history,f)
f.close()
'''

'''print('\nEvaluation of model on train data:')
trainAcc=model.evaluate(x=pcaTrainrecons4,y=Output_train,batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(trainAcc)

print('\nEvaluation of model on validation data:')
valAcc=model.evaluate(x=Input_val_Dataset, y=Output_val, batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(valAcc)

print('\nEvaluation of model on test data:')
testAcc=model.evaluate(x=Input_test_Dataset, y=Output_test, batch_size=1, verbose="auto",callbacks=None)
print(model.metrics_names)
print(testAcc)

print('\nPredictions for test data:')
testProb=model.predict(Input_test_Dataset, batch_size=1, verbose="auto", callbacks=None)
pred=np.argmax(testProb,axis=1)
print(pred)

#confusionMatrix=confusion_matrix(Output_test, pred)
confusionMatrix=confusion_matrix(Output_test,pred)
#confusionMatrix=tf.math.confusion_matrix(Output_test, pred5)
print(confusionMatrix)
'''
n=3
plt.figure(figsize=(20,4))
for i in range(n):
    #displaying original image
    ax=plt.subplot(2,n,i+1)
    plt.imshow(xTrain[i].reshape(28,28),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #displaying reconstructed image
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(trainEncode[i].reshape(8,8),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax=plt.subplot(2,n,i+1+2*n)
    plt.imshow(trainReconstruct[i].reshape(8,8),cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


             


