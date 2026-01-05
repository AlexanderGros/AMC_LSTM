# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:20:23 2024

@author: Aurélien

Helper file with the Datagenerator,
                       IQ to Polar converter,
                       SNR graph plotter and 
                       the Confusion matrix drawer
Still Work in Progress
"""
import numpy as np
import keras
import keras.models as models
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
#%%Confusion Matrix
def ConfDrawer(model,filepath,vec_len,Polar,conf_batch_size,n_examples):
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
    yval=y[test_idx]
    yhatval=yhat[test_idx]
    result = confusion_matrix(ynum, yhat , normalize='pred')
    print(result)
    disp = ConfusionMatrixDisplay(confusion_matrix=result,
                                   display_labels=possible_mods)
    plt.title("Confusion Matrix of All Data")
    disp.plot()
    plt.savefig(filepath+"_Val")
#%%
def SNRPlotter(SNR_step,model,filepath,vec_len,Polar,n_examples):
    h5fr = h5py.File('/CECI/trsf/umons/eletel/agros/spooner_full.h5','r')
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
    plt.savefig(filepath+"_SNR")
