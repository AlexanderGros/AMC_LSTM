# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:29:44 2024

@author: Aurélien

Two-layered LSTM network applied on spooner dataset


"""

import numpy as np

import matplotlib.pyplot as plt


import keras
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation

from tensorflow.keras.layers import LSTM


#from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
import keras_tuner

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import time

import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#%%


def length_selector(data,length):
    output = data[:,:,:length]
    return output

#vec_len = 4096   # 32768 Hyperparamètre
batch_size = 256 # -> to much mem for 1024
# LSTM_units1=128
# LSTM_units2=64
nb_epoch = 20   # number of epochs to train on (orig 100)
pate = 5
dirpath = '/home/users/a/n/aniebes/'
filepath = 'spooner_Grid_LSTM_model_Hypervec_len'
# Polar=True
LR=0.001
ConvModel=False

num_models=6
conf_batch_size = 256
SNR_step=0.5
#X = length_selector(X, vec_len)  # remove other file afterwards for memory optimization
print(filepath)
if ConvModel:
    print("Conventionnal model")
else:
    print("Alternative model")
print("vec_len: hyperparamètre")
print("LSTM_units: vec_len")
print("Polar: Polar")
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
#TODO: tester: class DataGenerator(keras.utils.PyDataset):
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
    if(ConvModel):
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
class MyHyperModel(keras_tuner.HyperModel):

    def build(self,hp):
        
        self.vec_len=hp.Int("vec_len", min_value=2250, max_value=3250, step=250)#256, 512, 1024, 2048
        LSTM_units1=self.vec_len
        LSTM_units2=self.vec_len
        if(ConvModel):
            in_shp=(self.vec_len, 2)
        else:
            in_shp=(2, self.vec_len)
        #in_shp = list([self.vec_len,2])   # This is the input shape of 2 channels x 128 time samples
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
                      metrics=["accuracy"])
    
        model.summary()
        return model
    def fit(self, hp, model, train_idx,y,batch_size, **kwargs):
        hp.Boolean("Polar")
        return model.fit(
            DataGenerator(train_idx, y, batch_size, self.vec_len, shuffle=True,polar=Polar),
            verbose=1,
            steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
            #show_accuracy=False,
            validation_data=DataGenerator(test_idx, y, batch_size, self.vec_len, shuffle=True,polar=Polar),
            **kwargs,
        )

# hp = keras_tuner.HyperParameters()
# hypermodel = MyHyperModel()
# model = hypermodel.build(hp)
# hypermodel.fit(hp, model, train_idx,y,batch_size,Polar,nb_epoch)
#%%
#filepath = 'spooner_iq_cnn_model_2048.wts.h5'
# print(' ')
# print('Reloading previous model')
# # we re-load the best weights once training is finished
# #model.load_weights(filepath)
# model = keras.models.load_model(filepath)
# print('end of loading')

#%%
tuner = keras_tuner.GridSearch(
    hypermodel=MyHyperModel(),
    objective="val_loss",
    executions_per_trial=1,
    overwrite=True,
    directory=dirpath,
    project_name=filepath,

)

tuner.search_space_summary()

#%%
print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset
history=tuner.search(train_idx,y,batch_size, epochs=nb_epoch,   callbacks = [
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
            ])
# history = model.fit(
#     DataGenerator(train_idx, y, batch_size, vec_len, shuffle=True,polar=Polar),
#     verbose=1,
#     steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
#     epochs=nb_epoch,
#     #show_accuracy=False,
#     validation_data=DataGenerator(test_idx, y, batch_size, vec_len, shuffle=True,polar=Polar),
#     callbacks = [
#         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
#         keras.callbacks.EarlyStopping(monitor='val_loss', patience=pate, verbose=1, mode='auto')
#     ])


print("training time:  %s seconds " % (time.time() - start_time))  # 

print('end of training')
#%%

models = tuner.get_best_models(num_models=num_models)
best_hp = tuner.get_best_hyperparameters(num_models)
best_model = models[0]
best_model.summary()

tuner.results_summary()

# #%%
# # plot loss during training
# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# # plot accuracy during training
# plt.subplot(212)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()
# plt.savefig(filepath+"_History")
#%%

#print(' ')
#print('Loading model')
## we re-load the best weights once training is finished
##model.load_weights(filepath)
#model = keras.models.load_model(filepath)
#print('end of loading')


#%%
# Show simple version of performance
#score = best_model.evaluate(DataGenerator(test_idx, y, batch_size=64, vec_len=vec_len, shuffle=True,polar=Polar),
#                       verbose=2,#Verbose=1: Progress bar, Verbose=2: single line
#                       return_dict=True)#score est un dict 
#print('end of evaluation')
#print(' ')
#print('score: ', score)# valeur du loss et des metrics Actuellement: score: [categorical_crossentropy, accuracy]

#%%Confusion Matrix
for m in range(num_models):
    model=models[m]
    hp=best_hp[m]
    
    vec_len=int(hp.get("vec_len"))
    Polar=hp.get("Polar")
    print(m,"rd Best model")
    print("Vec_len=",vec_len)
    print("Polar=",Polar)
    
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
        if ConvModel:
            batch=np.swapaxes(batch, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)#TODO: Réorganiser la bdd

        #Predict
        
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
    
    Confile=filepath
    if Polar:
        Confile=Confile+"_Polar"
    else:
        Confile=Confile+"_IQ"
    Confile=Confile+"_"+str(vec_len)
    plt.savefig(Confile)
    
    #On validation data
    ynum2=np.array(ynum)
    yval=ynum2[test_idx]
    yhatval=yhat[test_idx]
    result = confusion_matrix(yval, yhatval , normalize='pred')
    print(result)
    disp = ConfusionMatrixDisplay(confusion_matrix=result,
                                   display_labels=possible_mods)
    plt.title("Confusion Matrix of All Data")
    disp.plot()
    plt.savefig(Confile+"_Val")
    #%%
    print("Depending on the SNR")
    snrs=np.array(SNRS)
    onlyVal=False
    if(onlyVal):
        snrs=snrs[test_idx]
    for s in np.arange(-3,13,1):
        print("SNR",s)
        indexe_s=np.where(np.abs(snrs-s)<1/2)
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
              Convpath=Confile+"_Val_SNR"+str(s)+"dB"
              plt.title("Confusion Matrix of Validation Data (SNR="+str(s)+")")
            else:
                Convpath=Confile+"_SNR"+str(s)+"dB"
                plt.title("Confusion Matrix of All Data (SNR="+str(s)+")")
            plt.savefig(Convpath)
        else:
            print("Too little results")
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
            
            if ConvModel:
                batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)

            y_prediction = model.predict(batch_x)
            #print(np.shape(y_prediction))
            y_test=np.argmax(y_prediction, axis=1)
            #print(np.shape(y_test))
            #print(y_test)
            
            ynum2=np.array(ynum)
            trueY=ynum2[index_test_X_i]
            
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
#%%
'''
06/03/24 IQ 4096
Epoch 1/4
1225/1225 [==============================] - 6067s 5s/step - loss: 0.3455 - accuracy: 0.8895 - categorical_accuracy: 0.8895 - val_loss: 0.2802 - val_accuracy: 0.9133 - val_categorical_accuracy: 0.9133
Epoch 00001: val_loss improved from inf to 0.28016, saving model to /home/users/a/n/aniebes/spooner_iq_LSTM_model_4096
Epoch 2/4
1225/1225 [==============================] - 6058s 5s/step - loss: 0.1500 - accuracy: 0.9500 - categorical_accuracy: 0.9500 - val_loss: 0.3296 - val_accuracy: 0.9048 - val_categorical_accuracy: 0.9048
Epoch 00002: val_loss did not improve from 0.28016
Epoch 3/4
1225/1225 [==============================] - 5896s 5s/step - loss: 0.1874 - accuracy: 0.9383 - categorical_accuracy: 0.9383 - val_loss: 0.3450 - val_accuracy: 0.9015 - val_categorical_accuracy: 0.9015
Epoch 00003: val_loss did not improve from 0.28016
Epoch 4/4
1225/1225 [==============================] - 5842s 5s/step - loss: 0.1672 - accuracy: 0.9443 - categorical_accuracy: 0.9443 - val_loss: 0.3426 - val_accuracy: 0.9041 - val_categorical_accuracy: 0.9041
Epoch 00004: val_loss did not improve from 0.28016
training time:  23888.49215197563 seconds 
end of training
 
Loading model
end of loading
525/525 - 651s - loss: 0.2802 - accuracy: 0.9133 - categorical_accuracy: 0.9133
end of evaluation
 
score:  {'loss': 0.28016090393066406, 'accuracy': 0.9133333563804626, 'categorical_accuracy': 0.9133333563804626}
job end at Wed Mar  6 02:36:34 CET 2024
'''

'''

Training start
08/03/24 IQ 4096
Epoch 1/10
1225/1225 [==============================] - 5469s 4s/step - loss: 1.8211 - accuracy: 0.3471 - val_loss: 0.4837 - val_accuracy: 0.8516
2024-03-07 23:02:17.814755: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
Epoch 00001: val_loss improved from inf to 0.48374, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096 (note: model accidentally overwritten by iq 512)
Epoch 2/10
1225/1225 [==============================] - 5495s 4s/step - loss: 0.2814 - accuracy: 0.9109 - val_loss: 0.3116 - val_accuracy: 0.9042
Epoch 00002: val_loss improved from 0.48374 to 0.31165, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 3/10
1225/1225 [==============================] - 5514s 5s/step - loss: 0.1773 - accuracy: 0.9410 - val_loss: 0.3125 - val_accuracy: 0.9083
Epoch 00003: val_loss did not improve from 0.31165
Epoch 4/10
1225/1225 [==============================] - 5504s 4s/step - loss: 0.1510 - accuracy: 0.9501 - val_loss: 0.3241 - val_accuracy: 0.9069
Epoch 00004: val_loss did not improve from 0.31165
Epoch 5/10
1225/1225 [==============================] - 5519s 5s/step - loss: 0.1385 - accuracy: 0.9529 - val_loss: 0.3250 - val_accuracy: 0.9101
Epoch 00005: val_loss did not improve from 0.31165
Epoch 6/10
1225/1225 [==============================] - 5520s 5s/step - loss: 0.1307 - accuracy: 0.9566 - val_loss: 0.3348 - val_accuracy: 0.9096
Epoch 00006: val_loss did not improve from 0.31165
Epoch 7/10
1225/1225 [==============================] - 5517s 5s/step - loss: 0.1283 - accuracy: 0.9577 - val_loss: 0.3390 - val_accuracy: 0.9086
Epoch 00007: val_loss did not improve from 0.31165
Epoch 00007: early stopping
job end at Fri Mar  8 08:14:39 CET 2024
'''

'''
08/03/24 IQ 512
job start at Fri Mar  8 04:16:47 CET 2024
Epoch 1/20
1225/1225 [==============================] - 994s 807ms/step - loss: 1.9646 - accuracy: 0.2239 - auc: 0.6221 - val_loss: 1.5597 - val_accuracy: 0.3991 - val_auc: 0.8005
2024-03-08 04:34:08.228639: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
Epoch 00001: val_loss improved from inf to 1.55973, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096 (Note: renamed after in spooner_iq_LSTM_model_512)
Epoch 2/20
1225/1225 [==============================] - 85s 70ms/step - loss: 1.4425 - accuracy: 0.4543 - auc: 0.8360 - val_loss: 1.2795 - val_accuracy: 0.5393 - val_auc: 0.8808
Epoch 00002: val_loss improved from 1.55973 to 1.27950, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 3/20
1225/1225 [==============================] - 85s 69ms/step - loss: 1.0391 - accuracy: 0.6287 - auc: 0.9219 - val_loss: 1.0536 - val_accuracy: 0.6298 - val_auc: 0.9200
Epoch 00003: val_loss improved from 1.27950 to 1.05356, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 4/20
1225/1225 [==============================] - 85s 69ms/step - loss: 0.7291 - accuracy: 0.7441 - auc: 0.9615 - val_loss: 0.9281 - val_accuracy: 0.6777 - val_auc: 0.9370
Epoch 00004: val_loss improved from 1.05356 to 0.92814, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 5/20
225/1225 [==============================] - 85s 70ms/step - loss: 0.5470 - accuracy: 0.8112 - auc: 0.9778 - val_loss: 0.8574 - val_accuracy: 0.7058 - val_auc: 0.9458
Epoch 00005: val_loss improved from 0.92814 to 0.85735, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 6/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.4333 - accuracy: 0.8522 - auc: 0.9854 - val_loss: 0.8304 - val_accuracy: 0.7189 - val_auc: 0.9494
Epoch 00006: val_loss improved from 0.85735 to 0.83045, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 7/20
1225/1225 [==============================] - 85s 69ms/step - loss: 0.3698 - accuracy: 0.8712 - auc: 0.9892 - val_loss: 0.8292 - val_accuracy: 0.7303 - val_auc: 0.9499
Epoch 00007: val_loss improved from 0.83045 to 0.82922, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 8/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.3305 - accuracy: 0.8862 - auc: 0.9911 - val_loss: 0.8159 - val_accuracy: 0.7334 - val_auc: 0.9516
Epoch 00008: val_loss improved from 0.82922 to 0.81588, saving model to /home/users/a/n/aniebes/spooner_Polar_LSTM_model_4096
Epoch 9/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.3116 - accuracy: 0.8918 - auc: 0.9918 - val_loss: 0.8368 - val_accuracy: 0.7344 - val_auc: 0.9506
Epoch 00009: val_loss did not improve from 0.81588
Epoch 10/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.2755 - accuracy: 0.9038 - auc: 0.9934 - val_loss: 0.8366 - val_accuracy: 0.7373 - val_auc: 0.9512
Epoch 00010: val_loss did not improve from 0.81588
Epoch 11/20
1225/1225 [==============================] - 85s 69ms/step - loss: 0.2636 - accuracy: 0.9097 - auc: 0.9937 - val_loss: 0.8628 - val_accuracy: 0.7324 - val_auc: 0.9493
Epoch 00011: val_loss did not improve from 0.81588
Epoch 12/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.2452 - accuracy: 0.9139 - auc: 0.9947 - val_loss: 0.8557 - val_accuracy: 0.7366 - val_auc: 0.9501
Epoch 00012: val_loss did not improve from 0.81588
Epoch 13/20
1225/1225 [==============================] - 85s 70ms/step - loss: 0.2332 - accuracy: 0.9195 - auc: 0.9948 - val_loss: 0.8565 - val_accuracy: 0.7397 - val_auc: 0.9504
Epoch 00013: val_loss did not improve from 0.81588
Epoch 00013: early stopping
training time:  2146.5353350639343 seconds 
end of training
Loading model
end of loading
525/525 - 11s - loss: 0.8159 - accuracy: 0.7334 - auc: 0.9516
end of evaluation
 
score:  {'loss': 0.8158758282661438, 'accuracy': 0.7333630919456482, 'auc': 0.9516393542289734}
job end at Fri Mar  8 04:53:48 CET 2024

'''