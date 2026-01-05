# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:22:36 2024

Compute and plot the classification accuracy versus the SNR 
for one or multiple models

@author: Aurélien
"""

import matplotlib.pyplot as plt

import keras
import numpy as np
import pandas as pd
import h5py

Figname="Vec_len 1024"
n_examples = 112000
vec_lens=[1024,1024]
#batch_size=128
Polars=[True, False]
CorrModels=[False, False]
dirpath='/home/users/a/n/aniebes/'
filepaths = ['spooner_Polar_LSTM_model_1024',
             'spooner_IQ_LSTM_model_1024_units1024']
descs=["Alt Polar U1=U2=1024",
       "Alt IQ U1=U2=1024"]
step=0.5

#df = pd.read_csv('signal_param_all.csv', sep=',')
df = pd.read_csv('/CECI/trsf/umons/eletel/agros/signal_param_all.csv', sep=',')
mods = df['modulation']
mods = mods.tolist()

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


#%%
possible_mods = ['bpsk', 'qpsk', 'dqpsk', '8psk', 'msk', '16qam', '64qam', '256qam']    # there are 8 modulations in the spooner dataset
y = np.array(list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples))))
#y = to_onehot(y)
#y=y[:n_examples]

SNRS = df['snr']
SNRS = SNRS.tolist()
snrs=set(SNRS)
snrs=sorted(snrs)
SNRS= np.array(SNRS)

h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')
#%%
fig, ax = plt.subplots()
plt.ylim((0,1))
for m in range(len(filepaths)):
    filepath=filepaths[m]
    vec_len=vec_lens[m]
    Polar=Polars[m]
    desc=descs[m]
    CorrModel=CorrModels[m]
    
    print(filepath)
    print("vec_len:",vec_len)
    if Polar:
        print("Polar")
    else:
        print("IQ")
    if CorrModel:
        print("Conventionnal model")
    else:
        print("Alternative model")
    print("desc,",desc)
    
    print(' ')
    print('Reloading previous model')
    # we re-load the best weights once training is finished
    #model.load_weights(filepath)
    model = keras.models.load_model(filepath)
    print('end of loading')
    
    details=np.empty((0,3))
    results=np.empty((0,3))
    for s in np.arange(-3,13,step):
        #print(s)
        TotPop=0
        TotCorr=0
        index_test_X_i = np.argwhere(np.abs(SNRS-s)<=step)
        #print(np.size(index_test_X_i))
        #print(index_test_X_i)
        if(np.size(index_test_X_i)>10):
            index_test_X_i = index_test_X_i[:,0]
            batch_x = h5fr['spooner'][index_test_X_i,:,:vec_len]
            if(Polar):
                batch_x=to_amp_phase(batch_x,vec_len)
            if(CorrModel):
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
        
    
    ax.plot(results[:,0],results[:,1],"x-",label=desc)
    print(results)
ax.set_xlabel('SNR [dB]')
ax.set_ylabel('Validation accuracy')
plt.grid()
plt.legend()
plt.grid()
#
plt.savefig(Figname+"_SNR")


