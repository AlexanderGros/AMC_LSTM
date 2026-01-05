# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:29:44 2024

@author: Aurélien

Two-layered LSTM network applied on spooner dataset


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

import time

import h5py

from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from tensorflow.keras import models
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import json


# import keras_tuner as kt


from tensorflow.keras.models import load_model

from tensorflow.keras.utils import plot_model

import tensorflow as tf

from tensorflow.keras.layers import LSTM



import time

import h5py
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#%%


def length_selector(data,length):
    output = data[:,:,:length]
    return output
    
    
#%%!!! to complete 
vec_len = 128   # 32768   128
batch_size =128  # -> to much mem for 1024
LSTM_units1=128 #Length of the hidden and cell state
LSTM_units2=128
nb_epoch = 30   # number of epochs to train on (orig 100)
pate = 5
filepath = 'lstm_basic_no_cfo_128_128_128.keras'         #LSTM1 LSTM2 Samples 
ModelCorr=False # "Conventionnal" model      # to adapt !
Polar=True
LR=0.001
conf_batch_size = 256
SNR_step=0.5
Conv_SNR_step=1
#X = length_selector(X, vec_len)  # remove other file afterwards for memory optimization
print(filepath)
if ModelCorr:
    print("Conventionnal model")
else:
    print("Alternative model")
print("vec_len:",vec_len)
print("LSTM_units:",LSTM_units1,LSTM_units2)
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

#df = pd.read_csv('/home/umons/eletel/agros/trsf/signal_param_all_nf.csv', sep=',')  # /home/umons/eletel/agros/trsf/signal_param_all.csv
df = pd.read_csv('/home/umons/eletel/agros/trsf/signal_param_all_nf.csv', delim_whitespace=True)  #issue with cfo as space beween x 10^-3


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
    
    
    
def select_filtered_rows(df, nbr, mod=None, ups=None, downs=None, snr=None, span=10):

    mask = pd.Series(True, index=df.index)

    if mod is not None:
        mask &= (df['modulation'] == mod)
    if ups is not None:
        mask &= (df['upsample_factor'] == ups)
    if downs is not None:
        mask &= (df['downsample_factor'] == downs)
    if snr is not None:
        mask &= (df['snr'] >= snr)
    
    #return df[mask].head(nbr) #full df
    return df[mask].head(nbr).index.tolist() #list of indexes of selection
    
    
def correct_cfo(x, f_offset):
    n = np.arange(len(x))
    correction = np.exp(-1j * 2 * np.pi * f_offset * n )#/ sps)
    return x * correction
    

#%%
# generator

class DataGenerator(tf.keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)
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
    self.h5fr = h5py.File('/home/umons/eletel/agros/trsf/spooner_NF/spooner_full_NF.h5','r')   # /home/umons/eletel/agros/trsf/spooner_full_NF.h5
    

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
    if(ModelCorr):
        batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)
    
    # Assuming batch_x shape is (batch_size, 2, vec_len)
    complex_signal = batch_x[:,0,:] + 1j * batch_x[:,1,:]  # I + jQ
    
    
    rows = df.loc[list_IDs_temp]
    f_offsets = rows['carrier_offset'].values
    #T0 = rows['base_symbol_period'].values
    #U = rows['upsample_factor'].values
    #D = rows['downsample_factor'].values
    #sps = np.array([compute_sps(T0[i], U[i], D[i]) for i in range(len(list_IDs_temp))])
    
    # Now call correct_cfo with sps
    corrected_signal = np.array([
        correct_cfo2(complex_signal[i], f_offsets[i]) #, sps[i])
        for i in range(len(list_IDs_temp))
    ])
    '''
    # Apply CFO correction per sample
    rows = df.loc[list_IDs_temp]
    f_offsets = rows['carrier_offset'].values
    
    corrected_signal = np.array([
        correct_cfo(complex_signal[i], f_offsets[i])
        for i in range(len(list_IDs_temp))
    ])
    '''
    
    # Split back into I and Q for AI input
    batch_x = np.zeros_like(batch_x)
    batch_x[:,0,:] = np.real(corrected_signal)
    batch_x[:,1,:] = np.imag(corrected_signal)

    
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


if(ModelCorr):
    in_shp = tuple([vec_len, 2])   # This is the input shape of 2 channels x 128 time samples
else:
    in_shp = tuple([2, vec_len])
print (in_shp)

dr = 0.5 # dropout rate (%) = percentage of neurons to randomly lose each iteration
model = models.Sequential()  # Neural network is a set of sequential layers

model.add(LSTM(units=LSTM_units1, input_shape=in_shp, return_sequences=True , name="lstm1"))#return_sequences=True to output all the output sequence to the next layer

model.add(Dropout(dr))

model.add(LSTM(units=LSTM_units2, return_sequences=False, dropout=0.2, name="lstm2"))

#model.add(Dense(128, kernel_initializer="he_normal", activation="relu", name="dense2"))  #128   | 256

#model.add(Dropout(dr))

model.add(Dense( np.shape(classes)[0], kernel_initializer='he_normal', name="dense3" ))

model.add(Activation('softmax'))#Les outputs sont des probabilités dont la somme vaut 1
model.add(Reshape([np.shape(classes)[0]]))

opti=Adam(learning_rate=LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opti,
              metrics=['categorical_accuracy'])
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


import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.signal import upfirdn

# --- Helper Functions ---
def plot_constellation(I, Q, title, filename=None):
    plt.figure(figsize=(6,6))
    plt.scatter(I, Q, s=5)
    plt.title(title)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

def rrc_filter(beta, span, sps):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(t[i]) == 1/(4*beta):
            h[i] = (beta / np.sqrt(2) *
                    ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                     (1 - 2/np.pi) * np.cos(np.pi/(4*beta))))
        else:
            h[i] = (np.sin(np.pi*t[i]*(1-beta)) +
                    4*beta*t[i]*np.cos(np.pi*t[i]*(1+beta))) / \
                   (np.pi*t[i] * (1 - (4*beta*t[i])**2))
    h /= np.sqrt(np.sum(h**2))
    return h

def correct_cfo(x, f_offset, sps):
    n = np.arange(len(x))
    correction = np.exp(-1j * 2 * np.pi * f_offset * n / sps)
    return x * correction

def correct_cfo2(x, f_offset):
    n = np.arange(len(x))
    correction = np.exp(-1j * 2 * np.pi * f_offset * n )
    return x * correction
    
    
def compute_sps(T0, U, D):
    return U / D * T0  # Adjust as per your actual formula

def extract_symbols(x, sps, start):
    return x[start::sps]
    
    
# manual pipeline
def manual_pipeline(df, idx, vec_len):
    row = df.loc[idx]
    T0 = row['base_symbol_period']
    U, D = row['upsample_factor'], row['downsample_factor']
    sps = compute_sps(T0, U, D)
    f_offset = row['carrier_offset']
    print('manual f_offset: ',  f_offset)

    # Load raw data from the same file as the generator
    with h5py.File('/home/umons/eletel/agros/trsf/spooner_NF/spooner_full_NF.h5', 'r') as h5fr:
        xr = h5fr['spooner'][idx, 0, :vec_len]
        xi = h5fr['spooner'][idx, 1, :vec_len]
    x = xr + 1j * xi

    # CFO correction
    x_cfo_corrected = correct_cfo2(x, f_offset)
    plot_constellation(np.real(x_cfo_corrected), np.imag(x_cfo_corrected),
                       title='Manual: CFO Corrected', filename='manual_cfo.png')

    return x_cfo_corrected

# --- Main ---
# Select a row using your criteria
indexes = select_filtered_rows(df, nbr=1, mod='64qam', ups=6, downs=5, snr=2)
print('Selected indexes:', indexes)  # Debug: Print the selected indexes
idx = indexes[0]

# Manual pipeline
manual_cfo = manual_pipeline(df, idx, vec_len)

# Generator pipeline
gen = DataGenerator([idx], y, batch_size=1, vec_len=vec_len, polar=False)
batch_x, _ = gen[0]
gen_cfo = batch_x[0, 0, :] + 1j * batch_x[0, 1, :]
plot_constellation(np.real(gen_cfo), np.imag(gen_cfo),
                   title='Generator: CFO Corrected', filename='gen_cfo.png')

# Compare
print("Manual CFO (first 10 samples):", manual_cfo[:10])
print("Generator CFO (first 10 samples):", gen_cfo[:10])


# --- Generator Pipeline ---
def generator_pipeline(data_gen, idx, vec_len):
    batch_x, _ = data_gen[0]
    complex_gen = batch_x[0, 0, :] + 1j * batch_x[0, 1, :]
    plot_constellation(np.real(complex_gen), np.imag(complex_gen),
                       title='Generator: CFO Corrected', filename='gen_cfo.png')
    return complex_gen

# --- Main ---
'''
# Select a row using your criteria
indexes = select_filtered_rows(df, nbr=1, mod='64qam', ups=6, downs=5, snr=2)
print('Selected indexes:', indexes)  # Debug: Print the selected indexes
# Now you can safely use indexes[0]
idx = indexes[0]
'''

# Manual pipeline
manual_cfo = manual_pipeline(df, idx, vec_len)

# Generator pipeline
gen = DataGenerator([idx], y, batch_size=1, vec_len=vec_len, polar=False)
batch_x, _ = gen[0]
gen_cfo = batch_x[0, 0, :] + 1j * batch_x[0, 1, :]
plot_constellation(np.real(gen_cfo), np.imag(gen_cfo),
                   title='Generator: CFO Corrected', filename='gen_cfo.png')

# Compare
print("Manual CFO (first 10 samples):", manual_cfo[:10])
print("Generator CFO (first 10 samples):", gen_cfo[:10])







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
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
    ])


print("training time:  %s seconds " % (time.time() - start_time))  # 


# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('epochs_accuracy.png', bbox_inches='tight')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('epochs_loss_70.png', bbox_inches='tight')


print("training time all:  %s seconds " % (time.time() - start_time)) 


#saving training data tests: 
np.save('my_history_70.npy',history.history)

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
 
print('end of training')

'''
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
'''


  
#%%

print(' ')
print('Loading model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = tf.keras.models.load_model(filepath)
print('end of loading')




#%%

print(' ')
print('Evaluating model')

print('model.metrics_names: ', model.metrics_names)

# Show simple version of performance
score = model.evaluate(DataGenerator(test_idx, y, 32, vec_len, shuffle=False), verbose=1)   #for testing only first 100 ! -> change to all
print('end of evaluation')

#print(' ')
#scores.append(score)
print('score: ', score)




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
h5fr = h5py.File('/home/umons/eletel/agros/trsf/spooner_full.h5','r')
yhat=np.empty((n_examples))
for i in range(0,n_examples,conf_batch_size):
    batch=h5fr['spooner'][i:(i+conf_batch_size),:,:vec_len]
    #print(np.size(batch))
    if(Polar):
        batch=to_amp_phase(batch,vec_len)
    #Predict
    if(ModelCorr):
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
print("Depending on the SNR")
snrs=np.array(SNRS)
onlyVal=False
if(onlyVal):
    snrs=snrs[test_idx]
for s in np.arange(-3,13,Conv_SNR_step):
    print("SNR",s)
    indexe_s=np.where(np.abs(snrs-s)<Conv_SNR_step/2)
    print(np.size(indexe_s))
    if(np.size(indexe_s)>64):
        y_i = ynum2[indexe_s]
        yhat_i = yhat[indexe_s]
        
        result = confusion_matrix(y_i, yhat_i , normalize='pred')
        print(result)
        disp = ConfusionMatrixDisplay(confusion_matrix=result,
                                       display_labels=possible_mods)
        disp.plot()
        if(onlyVal):
          Convpath=filepath+"_Val_SNR"+str(s)+"dB"
          plt.title("Confusion Matrix of Validation Data (SNR="+str(s)+")")
        else:
            Convpath=filepath+"_SNR"+str(s)+"dB"
            plt.title("Confusion Matrix of All Data (SNR="+str(s)+")")
        plt.savefig(Convpath)
    else:
        print("Too little results")
#%%
print('Accuracy/SNR')
details=np.empty((0,3))
results=np.empty((0,3))

for s in np.arange(-3,13,SNR_step):
    #print(s)
    TotPop=0
    TotCorr=0
    index_test_X_i = np.argwhere(np.abs(SNRS-s)<=SNR_step)
    print(np.shape(index_test_X_i))
    #print(index_test_X_i)
    if(np.size(index_test_X_i)>10):
        index_test_X_i = index_test_X_i[:,0]
        batch_x = h5fr['spooner'][index_test_X_i,:,:vec_len]
        if(Polar):
            batch_x=to_amp_phase(batch_x,vec_len)
        if(ModelCorr):
            batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)#TODO: Réorganiser la bdd

        
        y_prediction = model.predict(batch_x)
        #print(np.shape(y_prediction))
        y_test=np.argmax(y_prediction, axis=1)
        #print(np.shape(y_test))
        #print(y_test)
        
        trueY=y[index_test_X_i]
        #print(np.shape(trueY))
        # m = keras.metrics.Accuracy()
        # m.update_state(trueY,y_test)
        # score=m.result()
        Corr=0
        Pop=0
        
        
        # for j in range(np.size(trueY)):
        #     if y_test[j]==trueY[j]:
        #         Corr+=1
        #     Pop+=1     
        Corr= np.size(np.nonzero(y_test==trueY))
        Pop=np.size(trueY)
        #print(snr,":",score)
        # score=np.insert(score,0,snr)
        # score=np.insert(score,0,np.size(index_test_X_i))
        # results=np.append(results,[score],0)
        details=np.append(details,[[s,Corr,Pop]],0)
        TotPop+=Pop
        TotCorr+=Corr
        #print(np.shape(results))
        #print(results[np.size(results,0)-1])
        if TotPop>0:
          Prec=TotCorr/TotPop
          res=[s,Prec,TotPop]
          print(res)
          results=np.append(results,[res],0)
          #print(results)
    
#print(results)
fig, ax = plt.subplots()
ax.plot(results[:,0],results[:,1],"bx-")
ax.set_xlabel('SNR [dB]')
ax.set_ylabel('Validation accuracy')
ax.grid()
plt.savefig(filepath+"_SNR")
#%%






