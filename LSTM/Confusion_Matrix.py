# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:52:02 2024

Compute and plot the confusion matrices of a model
for -all the dataset
    -the validation dataset
    -signals with given SNR
@author: Aurélien
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import keras
import numpy as np
import pandas as pd
import h5py

onlyVal=True

np.random.seed(2023)#To use only the validation data, the seed used for numpy.random should be the same than the one used when the model was trained
train_div=0.7

n_examples = 112000
ModelCorr=True
vec_len=128
batch_size=256
Polar=True
SNR_step=1
filepath = '/home/users/a/n/aniebes/spooner_Polar_LSTM_modelCorr_128_units32_128'

print(filepath)
print("vec_len:",vec_len)
if Polar:
    print("Polar")
else:
    print("IQ")
if onlyVal:
    print("Validation data")
else:
    print("All data")

#df = pd.read_csv('signal_param_all.csv', sep=',')
df = pd.read_csv('/CECI/trsf/umons/eletel/agros/signal_param_all.csv', sep=',')

mods = df['modulation']
mods = mods.tolist()

SNRS = df['snr']
SNRS = SNRS.tolist()

n_train = int(n_examples * train_div)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) 
test_idx = list(set(range(0,n_examples))-set(train_idx))

def to_onehot(yy):
    yy1 = np.zeros([np.shape(yy)[0], max(yy)+1]) 
    yy1[np.arange(np.shape(yy)[0]),yy] = 1
    return yy1
#%%
# to ampitude and phase conversion
# pay attention this also changes data shape

def to_amp_phase(X,nsamples):                # shapes
    #X (112000, 2, 32768)
    # train part
    X_cmplx = X[:,0,:] + 1j* X[:,1,:]   # (110000, 32768)
    
    X_amp = np.abs(X_cmplx)                   # (110000, 32768)
    X_ang = np.arctan2(X[:,1,:],X[:,0,:])/np.pi   # (110000, 32768)
    
    
    X_amp = np.reshape(X_amp,(-1,1,nsamples))  # (110000, 1, nsamples)
    X_ang = np.reshape(X_ang,(-1,1,nsamples))  # (110000, 1, nsamples)
    
    X = np.concatenate((X_amp,X_ang), axis=1)  # (110000, 2, nsamples)
    # comment next line if no transpose
    #X = np.transpose(np.array(X),(0,2,1))            # (110000, 128, 2)

    return X

possible_mods = ['bpsk', 'qpsk', 'dqpsk', '8psk', 'msk', '16qam', '64qam', '256qam']    # there are 8 modulations in the spooner dataset
y = np.array(list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples))))#
#y = to_onehot(y)
#y=y[:n_examples]
if(onlyVal):
    y=y[test_idx]

print(' ')
print('Reloading previous model')
# we re-load the best weights once training is finished
#model.load_weights(filepath)
model = keras.models.load_model(filepath)
print('end of loading')

h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')

print('Start of predictions')
yhat=np.empty((np.size(y)))
for i in range(0,np.size(y),batch_size):
    batch=h5fr['spooner'][i:(i+batch_size),:,:vec_len]
    if(onlyVal):
        batch=h5fr['spooner'][test_idx[i:(i+batch_size)],:,:vec_len]
    
    #print(np.size(batch))
    if(Polar):
        batch=to_amp_phase(batch,vec_len)
    if(ModelCorr):
        batch=np.swapaxes(batch, 1, -1)
    #Predict
    y_prediction = model.predict(batch)
    #print(np.size(y_prediction))
    y_test=np.argmax(y_prediction, axis=1)
    #print(np.size(y_test))
    yhat[i:(i+batch_size)]=y_test
print('Predictions end')
#Create confusion matrix and normalizes it over predicted (columns)
def plot_confusion_matrix(cm, title='BEMD Confusion matrix', cmap=plt.cm.Blues, labels=[], fontsize=40):
    plt.figure()
    plt.xticks(fontsize=30)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=15)
    plt.yticks(tick_marks, labels, fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    
conf = np.zeros([len(possible_mods),len(possible_mods)])
confnorm = np.zeros([len(possible_mods),len(possible_mods)])  # normalized version
# for i in range(0,n_examples):
#     j = list(Y_test[i,:]).index(1)  # true values
#     print(j)
#     k = int(np.argmax(test_Y_hat[i,:]))  # predicted values
#     print(k)
#     conf[j,k] = conf[j,k] + 1  # true-predicted
# for i in range(0,len(possible_mods)):
#     confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
result = confusion_matrix(y, yhat , normalize='pred')
print(result)
disp = ConfusionMatrixDisplay(confusion_matrix=result,
                               display_labels=possible_mods)
disp.plot()


if(onlyVal):
  Convpath=filepath+"_Val"
  plt.title("Confusion Matrix of Validation Data")
else:
    Convpath=filepath
    plt.title("Confusion Matrix of All Data")
plt.savefig(Convpath)

#%%
print("Depending on the SNR")
snrs=np.array(SNRS)
if(onlyVal):
    snrs=snrs[test_idx]
for s in np.arange(-3,13,SNR_step):
    print("SNR",s)
    indexe_s=np.where(np.abs(snrs-s)<SNR_step/2)
    print(np.size(indexe_s))
    if(np.size(indexe_s)>64):
        y_i = y[indexe_s]
        yhat_i = yhat[indexe_s]
        
        result = confusion_matrix(y_i, yhat_i , normalize='pred')
        print(result)
        disp = ConfusionMatrixDisplay(confusion_matrix=result,
                                       display_labels=possible_mods)
        disp.plot()
        if(onlyVal):
          Convpath=filepath+"_Val_SNR"+str(s)
          plt.title("Confusion Matrix of Validation Data (SNR="+str(s)+")")
        else:
            Convpath=filepath+"_SNR"+str(s)
            plt.title("Confusion Matrix of All Data (SNR="+str(s)+")")
        plt.savefig(Convpath)
    else:
        print("Too little results")
