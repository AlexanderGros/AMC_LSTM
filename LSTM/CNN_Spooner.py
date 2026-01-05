# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:34:43 2024

@author: Aurélien

Baseline CNN Model
"""
import numpy as np

import matplotlib.pyplot as plt


import keras
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation
from keras import regularizers

from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, Flatten


#from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import time

import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#%%


def length_selector(data,length):
    output = data[:,:,:length]
    return output

vec_len = 128   # 32768
batch_size = 128 # -> to much mem for 1024
nb_epoch = 100   # number of epochs to train on (orig 100)
pate = 20
filepath = '/home/users/a/n/aniebes/spooner_IQ_CNN_model_veclen128_Regul'
Polar=False
LR=0.001
conf_batch_size = 256
SNR_step=0.5
#X = length_selector(X, vec_len)  # remove other file afterwards for memory optimization
print(filepath)
print("vec_len:",vec_len)
if Polar:
    print("Polar")
else:
    print("IQ")
print("batch_size:",batch_size)
print("learning rate:", LR)


#print('shorted X data shape', np.shape(X))


#%%
# one hot encoding the spooner 

import pandas as pd

df = pd.read_csv('/CECI/trsf/umons/eletel/agros/signal_param_all.csv', sep=',')


mods = df['modulation']
mods = mods.tolist()


SNRS = df['snr']
SNRS = SNRS.tolist()
snrs=set(SNRS)
snrs=sorted(snrs)


#keras.utils.set_random_seed(2023)
np.random.seed(2023)
n_examples = 112000
n_train = int(n_examples * 0.7) # Why a 50/50 split instead of 60/40 or 70/30??  yeah good question

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) 
test_idx = list(set(range(0,n_examples))-set(train_idx))
#X_train = X[train_idx]
#X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([np.shape(yy)[0], max(yy)+1]) 
    yy1[np.arange(np.shape(yy)[0]),yy] = 1
    return yy1

possible_mods = ['bpsk', 'qpsk', 'dqpsk', '8psk', 'msk', '16qam', '64qam', '256qam']    # there are 8 modulations in the spooner dataset

#Y_train = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), train_idx))) # only get the modulation

#Y_test = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), test_idx)))

ynum = list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples)))
y=to_onehot(ynum)

# defining classes


classes = possible_mods
#%%
# to ampitude and phase conversion
# pay attention this also changes data shape

def to_amp_phase(X,nsamples):                # shapes
    #X (112000, 2, 32768)
    # train part
    X_cmplx = X[:,0,:] + 1j* X[:,1,:]   # (110000, 32768)
    
    X_amp = np.abs(X_cmplx)                   # (110000, 32768)
    X_ang = np.arctan2(X[:,1,:],X[:,0,:])/np.pi   # (110000, 32768)
    #X_ang=np.angle(X_cmplx)/np.pi
    
    
    X_amp = np.reshape(X_amp,(-1,1,nsamples))  # (110000, 1, nsamples)
    X_ang = np.reshape(X_ang,(-1,1,nsamples))  # (110000, 1, nsamples)
    
    X = np.concatenate((X_amp,X_ang), axis=1)  # (110000, 2, nsamples)
    # comment next line if no transpose
    #X = np.transpose(np.array(X),(0,2,1))            # (110000, 128, 2)

    return X

#%%
# generator

class DataGenerator(keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)
  def __init__(self, list_IDs, labels, batch_size, vec_len, shuffle=True, polar=False, **kwargs):
    super().__init__(**kwargs)
    self.list_IDs = list_IDs
    self.labels = labels
    self.batch_size = batch_size
    self.vec_len = vec_len
    self.shuffle = shuffle
    self.polar= polar
    
    #self.test_list=np.copy(self.list_IDs)
    
    self.on_epoch_end()
    self.h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')
    
    
    
    

  def __len__(self):
    return int(np.ceil(len(self.list_IDs) / self.batch_size))
	

  def __getitem__(self, index):
  
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    list_IDs_temp = sorted(list_IDs_temp)
    
    #Pour tester si le générateur sort bien toutes les données
    #self.test_list[indexes]=-1
    #print("Nouveau batch, plus que ",np.count_nonzero(self.test_list!=-1))
    
    # Retrieve data
    batch_x = self.h5fr['spooner'][list_IDs_temp,:,:self.vec_len]
    if(self.polar):#TODO: Convertir tout le dataset en polaire
        batch_x=to_amp_phase(batch_x,self.vec_len)
    batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)
    batch_y = self.labels[list_IDs_temp]                
    return batch_x, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    #print("Nouvelle epoch, il reste ",np.count_nonzero(self.test_list!=-1))#Pour tester si le générateur sort bien toutes les données
    
    self.indexes = np.arange(len(self.list_IDs))
    
    #Pour tester si le générateur sort bien toutes les données
    #self.test_list=np.copy(self.list_IDs)
    #print("Nouvelle epoch ",np.count_nonzero(self.test_list!=-1))
    
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')



#%%

print('start AI architecture definition')



in_shp = list([vec_len, 2])   # This is the input shape of 2 channels x 128 time samples
print (in_shp)

rw = 0.01 # regularizer weight 
model = models.Sequential()  # Neural network is a set of sequential layers

model.add(Input(shape=in_shp, name="Input"))
model.add(Reshape((2,vec_len)))
#model.add( Input(shape=(2,128,8)) )


#model.add(ZeroPadding2D((0, 2)))  # Add 2 columns of zeros to each side
model.add(Convolution2D(256, (1, 3), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L2(rw))) # 128  | 160


#model.add(ZeroPadding2D((0, 2))) # Add 2 columns of zeros to each side
model.add(Convolution2D(80, (2, 3), padding="same", activation="relu", name="conv2", kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L2(rw))) # 40  |  80

#model.add(Dropout(dr))

model.add(Flatten())

#model.add(Dense(1024, kernel_initializer="he_normal", activation="relu", name="dense1"))

#model.add(Dropout(dr))

model.add(Dense(256, kernel_initializer="he_normal", activation="relu", name="dense2"))  #128   | 256

#model.add(Dropout(dr))

model.add(Dense( np.shape(classes)[0], kernel_initializer='he_normal', name="dense3" ))

model.add(Activation('softmax'))
model.add(Reshape([np.shape(classes)[0]]))


opti=Adam(learning_rate=LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opti,
              metrics=["accuracy"])

model.summary()
#%%
##filepath = 'spooner_iq_cnn_model_2048.wts.h5'
# print(' ')
# print('Reloading previous model')
## we re-load the best weights once training is finished
##model.load_weights(filepath)
# model = keras.models.load_model(filepath)
# print('end of loading')

#%%

print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset

history = model.fit(
    DataGenerator(train_idx, y, batch_size, vec_len, shuffle=True,polar=Polar),
    verbose=1,
    steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
    epochs=nb_epoch,
    #show_accuracy=False,
    validation_data=DataGenerator(test_idx, y, batch_size, vec_len, shuffle=True,polar=Polar),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
    ])


print("training time:  %s seconds " % (time.time() - start_time))  # 

print('end of training')
#%%
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
plt.savefig(filepath+"_History")
#%%

print(' ')
print('Loading model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = keras.models.load_model(filepath)
print('end of loading')


#%%
# Show simple version of performance
score = model.evaluate(DataGenerator(test_idx, y, batch_size=64, vec_len=vec_len, shuffle=True,polar=Polar),
                       verbose=2,#Verbose=1: Progress bar, Verbose=2: single line
                       return_dict=True)#score est un dict 
print('end of evaluation')
print(' ')
print('score: ', score)# valeur du loss et des metrics Actuellement: score: [categorical_crossentropy, accuracy]

#%%Confusion Matrix
print("Confusion Matrix")
print('Start of predictions')
#TODO:Faire que ça fontionne quand n_examples n'est pas un multiple de conf_batch_size
h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')
yhat=np.empty((n_examples))
for i in range(0,n_examples,conf_batch_size):
    batch=h5fr['spooner'][i:(i+conf_batch_size),:,:vec_len]
    #print(np.size(batch))
    if(Polar):
        batch=to_amp_phase(batch,vec_len)
    #Predict
    batch=np.swapaxes(batch, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)

    y_prediction = model.predict(batch)
    #print(np.size(y_prediction))
    y_test=np.argmax(y_prediction, axis=1)
    #print(np.size(y_test))
    yhat[i:(i+conf_batch_size)]=y_test
print('Predictions end')
result = confusion_matrix(ynum, yhat , normalize='pred')
print(result)
disp = ConfusionMatrixDisplay(confusion_matrix=result,
                               display_labels=possible_mods)
plt.title("Confusion Matrix of All Data")
disp.plot()
plt.savefig(filepath)

#On validation data
ynum2=np.array(ynum)
yval=ynum2[test_idx]
yhatval=yhat[test_idx]
result = confusion_matrix(yval, yhatval , normalize='pred')
print(result)
disp = ConfusionMatrixDisplay(confusion_matrix=result,
                               display_labels=possible_mods)
plt.title("Confusion Matrix of Validation Data")
disp.plot()
plt.savefig(filepath+"_Val")
#%%
print('Accuracy/SNR')
details=np.empty((0,3))
results=np.empty((0,3))
i=0
for s in np.arange(-3,13,SNR_step):
    #print(s)
    TotPop=0
    TotCorr=0
    while True:
        snr=snrs[i]
        if abs(snr-s)>SNR_step/2:
          if TotPop>0:
            Prec=TotCorr/TotPop
            res=[s,Prec,TotPop]
            print(res)
            results=np.append(results,[res],0)
            #print(results)
          break;
            
        index_test_X_i = np.argwhere(np.array(SNRS)==snr)
        index_test_X_i = index_test_X_i[:,0]
        batch_x = h5fr['spooner'][index_test_X_i,:,:vec_len]
        if(Polar):
            batch_x=to_amp_phase(batch_x,vec_len)
            
        batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)

        
        y_prediction = model.predict(batch_x)
        #print(np.shape(y_prediction))
        y_test=np.argmax(y_prediction, axis=1)
        #print(np.shape(y_test))
        #print(y_test)
        
        ynum2=np.array(ynum)
        trueY=ynum2[index_test_X_i]
        #print(np.shape(trueY))
        # m = keras.metrics.Accuracy()
        # m.update_state(trueY,y_test)
        # score=m.result()
        Corr=0
        Pop=0
        for j in range(np.size(trueY)):
            if y_test[j]==trueY[j]:
                Corr+=1
            Pop+=1        
        #print(snr,":",score)
        # score=np.insert(score,0,snr)
        # score=np.insert(score,0,np.size(index_test_X_i))
        # results=np.append(results,[score],0)
        details=np.append(details,[[snr,Corr,Pop]],0)
        TotPop+=Pop
        TotCorr+=Corr
        #print(np.shape(results))
        #print(results[np.size(results,0)-1])
        i+=1
    
#print(results)
fig, ax = plt.subplots()
ax.plot(results[:,0],results[:,1],"bx-")
ax.set_xlabel('SNR [dB]')
ax.set_ylabel('Validation accuracy')
ax.grid()
plt.savefig(filepath+"_SNR")