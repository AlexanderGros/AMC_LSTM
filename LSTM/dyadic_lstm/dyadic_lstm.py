# -*- coding: utf-8 -*-
"""
Created on 2025

@author: Alexander

Interleaved ADC's mismatches creator
"""

#%%
# module imports

import numpy as np
import matplotlib.pyplot as plt

#from scipy.signal import ShortTimeFFT       #new version
from scipy.signal.windows import gaussian
from scipy.signal import stft, spectrogram
from scipy.signal import welch
from scipy.interpolate import interp1d


import sys, types


from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from tensorflow.keras import models
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


from tensorflow.keras.layers import LSTM


#from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

#import cv2

import scipy.signal
import pandas as pd

import time

import h5py


#%%
# Parameter definitions

fs_adc = 100e6  # Sampling rate of each ADC (100 MSPS)
fs_interleaved = 2 * fs_adc  # Interleaved sampling rate (200 MSPS)
f_in = 10e6  # Input signal frequency (10 MHz)
total_time = 10*1e-6  # Total time for simulation (1 microsecond)
n_samples = int(total_time * fs_interleaved)  # Total number of interleaved samples
print('total number of samples: ', n_samples)

# Time vectors for interleaved and individual ADCs
t_interleaved = np.arange(n_samples) / fs_interleaved
t_adc1 = t_interleaved[::2]  # ADC1 samples
t_adc2 = t_interleaved[1::2]  # ADC2 samples

# Input waveform
input_waveform = np.sin(2 * np.pi * f_in * t_interleaved)





#%%
# definitions



#%%
# 
'''
def length_selector(data,length):
    output = data[:,:,:length]
    return output
''' 

vec_len = 4096   # 256 1024  2048  4096   8192  16384   32768


#%%!!! to complete 
vec_len = 4096   # 32768
batch_size =128  # -> to much mem for 1024
LSTM_units1=64 #Length of the hidden and cell state
LSTM_units2=64
nb_epoch = 50   # number of epochs to train on (orig 100)
pate = 10
filepath = 'dyadic_lstm_1.keras'         #LSTM1 LSTM2 Samples 
ModelCorr=True  # "Conventionnal" model if true      # to adapt !
Polar=False
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




df = pd.read_csv('/home/umons/eletel/agros/trsf/signal_param_all.csv', sep=',')



mods = df['modulation']
np.shape(mods)
mods = mods.tolist()
type(mods)
print(' ')


snr = df['snr']
snr = snr.tolist()



np.random.seed(2023)
#n_examples = X.shape[0]
n_examples = 112000  #full spooner data size
n_train = int(n_examples * 0.7)  # Why a 50/50 split instead of 60/40 or 70/30??  yeah good question  !!! for spooner change to 70-30 !

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) 
# -> random selection for training (a shuffle is performed by keras !)
test_idx = list(set(range(0,n_examples))-set(train_idx))



def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1]) 
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

possible_mods = ['bpsk', 'qpsk', 'dqpsk', '8psk', 'msk', '16qam', '64qam', '256qam']    # there are 8 modulations in the spooner dataset

#Y_train = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), train_idx))) # only get the modulation

#Y_test = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), test_idx)))

y = to_onehot(list(map(lambda x: possible_mods.index(mods[x]), np.arange(n_examples))))


#print('first y_train shape: ', np.shape(Y_train))
print('first y shape: ', np.shape(y))


classes = possible_mods


#%%
# MM function

def simulate_passband_iq_mismatch(
    sig,
    carrier_freq,
    M=2,
    offsets_I=None,
    offsets_Q=None,
    gains_I=None,
    gains_Q=None,
    skews_I=None,
    skews_Q=None,
    normalize=False
):
    batch_size, _, N = sig.shape
    t = np.arange(N)
    
    # Initialize default mismatch parameters
    if offsets_I is None: offsets_I = [0.0] * M
    if offsets_Q is None: offsets_Q = [0.0] * M
    if gains_I   is None: gains_I   = [1.0] * M
    if gains_Q   is None: gains_Q   = [1.0] * M
    if skews_I   is None: skews_I   = [0.0] * M
    if skews_Q   is None: skews_Q   = [0.0] * M

    out = np.zeros_like(sig)

    for b in range(batch_size):
        I = sig[b, 0, :]
        Q = sig[b, 1, :]
        complex_baseband = I + 1j * Q

        # Full complex passband modulation
        passband = complex_baseband * np.exp(1j * 2 * np.pi * carrier_freq * t)

        # Interpolate real and imag parts separately
        interp_real = interp1d(t, np.real(passband), kind='cubic', fill_value='extrapolate')
        interp_imag = interp1d(t, np.imag(passband), kind='cubic', fill_value='extrapolate')

        # Create new passband signal with mismatch injected per interleaved channel
        passband_mm = np.zeros(N, dtype=np.complex64)

        for m in range(M):
            indices = np.arange(m, N, M)

            # Apply separate skews to I and Q components
            t_I_skewed = indices + skews_I[m]
            t_Q_skewed = indices + skews_Q[m]

            I_interp = interp_real(t_I_skewed)
            Q_interp = interp_imag(t_Q_skewed)

            # Apply gain and offset mismatches separately
            I_corr = gains_I[m] * I_interp + offsets_I[m]
            Q_corr = gains_Q[m] * Q_interp + offsets_Q[m]

            passband_mm[indices] = I_corr + 1j * Q_corr

        # Demodulate back to baseband
        downconverted = passband_mm * np.exp(-1j * 2 * np.pi * carrier_freq * t)
        out[b, 0, :] = np.real(downconverted)
        out[b, 1, :] = np.imag(downconverted)

    if normalize:
        out = normalize_per_waveform(out)

    return out


'''
def dyadic_filter(signal, decimation_factor):
    #signal has 3 dimensions
    filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal
'''

def dyadic_filter(signal, decimation_factor, ModelCorr):
    #signal has 3 dimensions
    if ModelCorr == True: #classic conventional model has swithched input
      filtered_signal = signal[:,::decimation_factor,:]
    else: #alternative lstm has classic input shape
      filtered_signal = signal[:,:,::decimation_factor]
    return filtered_signal
    
    

def normalize_per_waveform(x):
    '''
    Normalize each waveform in the batch to [-1, 1] range.
    x: shape (batch_size, 2, length)
    '''
    # Compute min and max across I & Q channels and time for each sample
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    
    # Normalize to [0, 1], then scale to [-1, 1]
    x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    return x_norm





'''
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size, vec_len, 
                 M=2, f_in=0.0, offsets_I=None, offsets_Q=None,
                 gains_I=None, gains_Q=None, skews_I=None, skews_Q=None,
                 normalize=False, shuffle=True):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.vec_len = vec_len
        self.shuffle = shuffle
        self.M = M
        self.f_in = f_in
        self.offsets_I = offsets_I
        self.offsets_Q = offsets_Q
        self.gains_I = gains_I
        self.gains_Q = gains_Q
        self.skews_I = skews_I
        self.skews_Q = skews_Q
        self.normalize = normalize
        self.on_epoch_end()
        self.h5fr = h5py.File('/home/umons/eletel/agros/trsf/spooner_full.h5','r')

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp = sorted(list_IDs_temp)
        
        X_batch = self.h5fr['spooner'][list_IDs_temp,:,:self.vec_len]
        
        batch_x_MM = simulate_passband_iq_mismatch(
        X_batch,
        carrier_freq=self.f_in,
        M=self.M,
        offsets_I=self.offsets_I,
        offsets_Q=self.offsets_Q,
        gains_I=self.gains_I,
        gains_Q=self.gains_Q,
        skews_I=self.skews_I,
        skews_Q=self.skews_Q,
        normalize=self.normalize
        )

        
        batch_x = ( batch_x_MM, dyadic_filter(batch_x_MM, 2), dyadic_filter(batch_x_MM, 4), dyadic_filter(batch_x_MM, 8), dyadic_filter(batch_x_MM, 16), 
               dyadic_filter(batch_x_MM, 32)  )

        batch_y = self.labels[list_IDs_temp]                
        return batch_x, batch_y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
'''

class DataGenerator(tf.keras.utils.Sequence):  # on   dragon2 (tested)    #DataGenerator('train', train_idx, Y_train, 32, vec_len, shuffle=True)

  def __init__(self, list_IDs, labels, batch_size, vec_len, polar=False, shuffle=True, **kwargs):
    super().__init__(**kwargs)
    self.list_IDs = list_IDs
    self.labels = labels
    self.batch_size = batch_size
    self.vec_len = vec_len
    self.shuffle = shuffle
    self.on_epoch_end()
    self.h5fr = h5py.File('/home/umons/eletel/agros/trsf/spooner_full.h5','r')
    self.polar= polar

  def __len__(self):
    return int(np.floor(len(self.list_IDs) / self.batch_size))
	

  def __getitem__(self, index):
  
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    list_IDs_temp = sorted(list_IDs_temp)
    
    
    # Retrieve data
               
    batch_x = self.h5fr['spooner'][list_IDs_temp,:,:self.vec_len]
    #print('bathx:', np.shape(batch_x))
    if(self.polar):#TODO: Convertir tout le dataset en polaire
        batch_x=to_amp_phase(batch_x,self.vec_len)
    if(ModelCorr): # for classic, conventional model the inputs are diferent than cnn, so we need to adapt
        batch_x=np.swapaxes(batch_x, 1, -1) #Pour que les dimensions des inputs du LSTM soient bien (batch_size, vec_len, n_features) et non (batch_size, n_features, vec_len)
    #print('modelcorr bathx:', np.shape(batch_x))
    #dyadic transform 
    X_batch = (batch_x, dyadic_filter(batch_x, 2, ModelCorr), dyadic_filter(batch_x, 4, ModelCorr), dyadic_filter(batch_x, 8, ModelCorr), dyadic_filter(batch_x, 16, ModelCorr), 
               dyadic_filter(batch_x, 32, ModelCorr))
               
    batch_y = self.labels[list_IDs_temp]                
    return X_batch, batch_y
    
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
            
            
print('datagenerator definition successfull')        


'''
# datagen test
# Instantiate your generator
gen = DataGenerator(
    list_IDs=train_idx,
    labels=y,
    batch_size=32,
    vec_len=vec_len,
    shuffle=True
)

# Get the first batch (index 0)
X_batch, y_batch = gen[0]

print("X_batch type:", type(X_batch))
print("X_batch shapes:", [x.shape for x in X_batch])  # Since X_batch is a list of arrays
print("y_batch shape:", y_batch.shape)
'''






#%%

# Set up some model params 
nb_epoch = 100   # number of epochs to train on (orig 100)
batch_size = 32  # training batch size
pate = 10



#%%
# architecture


print('start AI architecture definition')


'''
dr=0.5

tf.keras.backend.clear_session()


channel_1 = Input(shape=(2,vec_len, 1), name="input1")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_1)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_1 = Flatten()(x)

channel_2 = Input(shape=(2,vec_len//2, 1), name="input2")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_2)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_2 = Flatten()(x)

channel_3 = Input(shape=(2,vec_len//4, 1), name="input3")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_3)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_3 = Flatten()(x)

channel_4 = Input(shape=(2,vec_len//8, 1), name="input4")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_4)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_4 = Flatten()(x)

channel_5 = Input(shape=(2,vec_len//16, 1), name="input5")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_5)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_5 = Flatten()(x)

channel_6 = Input(shape=(2,vec_len//32, 1), name="input6")
x = Convolution2D(40, (2, 8), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(channel_6)
x = Dropout(dr)(x)
x = Convolution2D(10, (1, 4), padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
x = Dropout(dr)(x)
out_6 = Flatten()(x)



concatenated = concatenate([out_1, out_2, out_3, out_4, out_5, out_6])


out = Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1")(concatenated)

#out = Dropout(dr)(out)

out = Dense( len(classes), activation='softmax', kernel_initializer='he_normal', name="dense2")(out)

out = Reshape([len(classes)])(out)



model = Model(inputs = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6], outputs = out)


optim = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=["categorical_accuracy"])


model.summary()


print('end of architecture')
'''



'''

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

model.add(Activation('softmax'))#Les outputs sont des probabilites dont la somme vaut 1
model.add(Reshape([np.shape(classes)[0]]))

opti=Adam(learning_rate=LR)
model.compile(loss='categorical_crossentropy',
              optimizer=opti,
              metrics=['categorical_accuracy'])
              
'''

#configuration
dr = 0.5           # dropout rate
LR = 0.001        # learning rate


#input shape
if ModelCorr:
    in_shp = (vec_len, 2)  # (time_steps, features)
else:
    in_shp = (2, vec_len)  # (features, time_steps)
print("Input shape per channel:", in_shp)


def lstm_branch(x):
    x = LSTM(LSTM_units1, return_sequences=True)(x)
    x = Dropout(dr)(x)
    x = LSTM(LSTM_units2, return_sequences=False, dropout=0.2)(x)
    return x

#
channel_1 = Input(shape=in_shp, name="input1")
channel_2 = Input(shape=(in_shp[0]//2, in_shp[1]), name="input2")
channel_3 = Input(shape=(in_shp[0]//4, in_shp[1]), name="input3")
channel_4 = Input(shape=(in_shp[0]//8, in_shp[1]), name="input4")
channel_5 = Input(shape=(in_shp[0]//16, in_shp[1]), name="input5")
channel_6 = Input(shape=(in_shp[0]//32, in_shp[1]), name="input6")

# 
out_1 = lstm_branch(channel_1)
out_2 = lstm_branch(channel_2)
out_3 = lstm_branch(channel_3)
out_4 = lstm_branch(channel_4)
out_5 = lstm_branch(channel_5)
out_6 = lstm_branch(channel_6)

# 
merged = concatenate([out_1, out_2, out_3, out_4, out_5, out_6])

# 
x = Dense(256, activation="relu", kernel_initializer="he_normal", name="dense1")(merged)
x = Dense(len(classes), activation="softmax", kernel_initializer="he_normal", name="dense2")(x)
output = Reshape([len(classes)])(x)

# 
model = models.Model(
    inputs=[channel_1, channel_2, channel_3, channel_4, channel_5, channel_6],
    outputs=output
)

#
optim = Adam(learning_rate=LR)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optim,
    metrics=["categorical_accuracy"]
)

model.summary()




#%%

print(' ')
print('Training start')
start_time = time.time()


# perform training ...
#   - call the main training loop in keras for our network+dataset

history = model.fit(
    DataGenerator(train_idx, y, batch_size, vec_len, shuffle=True, polar=Polar),
    verbose=1,
    steps_per_epoch=len(train_idx)//batch_size, # replaces batch size
    epochs=nb_epoch,
    #show_accuracy=False,
    validation_data=DataGenerator(test_idx, y, batch_size, vec_len, shuffle=True, polar=Polar),
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











"""



#%%
# pretrained model filepath
filepath = './models/test.keras'


#%%




print(' ')
print('Loading model')
model = tf.keras.models.load_model(filepath)
print('end of loading')







#%%
#testing on more realistic data


import numpy as np

def generate_adc_mismatches(M, bits=8, full_scale=2.0, mode='moderate', seed=None):
    '''
    Generate realistic mismatch values for M interleaved ADC channels.

        Parameters:
        M (int): Number of interleaved ADC channels
        bits (int): ADC resolution (bits)
        full scale (float): Peak-to-peak full-scale voltage (2V for -1 to 1)
        mode (str): 'relaxed', 'moderate', or 'severe'
        seed (int): Optional random seed for reproducibility
        
        Returns:
        dict: {
            gain_factors: np.array of per-channel gains (around 1.0),
            offset_errors: np.array of DC offsets (V),
            timing_skews: np.array of normalized timing skews (fraction of Ts)
        }
        
'''
    
    if seed is not None:
        np.random.seed(seed)

    lsb = full_scale / (2 ** bits)

    # Choose s for each mode
    if mode == 'relaxed':
        gain_sigma   = 0.001   # 0.1% ? s for a gain factor centered at 1.0
        offset_sigma = 0.25 * lsb
        skew_sigma   = 0.0005  # 0.05% of Ts
    elif mode == 'nomm':
        gain_sigma   = 0.0   
        offset_sigma = 0.0 * lsb
        skew_sigma   = 0.0    
    elif mode == 'severe':
        gain_sigma   = 0.015   # 1.5%
        offset_sigma = 2.0 * lsb
        skew_sigma   = 0.01    # 1% of Ts
    elif mode == 'severe2':
        gain_sigma   = 0.05   # 5%
        offset_sigma = 5.0 * lsb
        skew_sigma   = 0.1    # 10% of Ts
    elif mode == 'extreme':
        gain_sigma   = 0.15   # 15%
        offset_sigma = 10.0 * lsb
        skew_sigma   = 0.2    # 20% of Ts
    else:  # 'moderate'
        gain_sigma   = 0.005   # 0.5%
        offset_sigma = 1.0 * lsb
        skew_sigma   = 0.003   # 0.3% of Ts

    # Draw actual gains around 1.0
    gain_factors  = np.random.normal(loc=1.0, scale=gain_sigma, size=M)
    offset_errors = np.random.normal(loc=0.0, scale=offset_sigma, size=M)
    timing_skews  = np.random.normal(loc=0.0, scale=skew_sigma, size=M)

    return {
        'gain_factors': gain_factors,
        'offset_errors': offset_errors,
        'timing_skews': timing_skews
    }




'''
mismatches = generate_adc_mismatches(M=4, mode='moderate', seed=42)
print("Gain factors   :", mismatches['gain_factors'])
print("Offsets (V)    :", mismatches['offset_errors'])
print("Skews (Ts frac):", mismatches['timing_skews'])
'''



def get_mismatches(M, mode='nomm', seed=None): # seed in other fct
    mismatches = generate_adc_mismatches(M=M, mode=mode, seed=seed)
    gains = mismatches['gain_factors'].tolist()
    offsets = mismatches['offset_errors'].tolist()
    skews = mismatches['timing_skews'].tolist()
    return gains, offsets, skews


modus='nomm'
print('modus: ', modus)
num_trials = 50  # orig10, changed to 50
results = {'M2': [], 'M4': []}



for M in [2, 4]:
    print(f"\n=== Evaluating M = {M} with severe mismatches ===")
    for trial in range(num_trials):
        print(f"\n-- Trial {trial + 1} --")
        seed = trial + 100  # different seed per trial

        # Generate mismatches
        gains, offsets, skews = get_mismatches(M=M, mode=modus, seed=seed)
        # for Q
        gains2, offsets2, skews2 = get_mismatches(M=M, mode=modus, seed=seed+1)
        
        trial_result = {}

        # GAIN only
        generator = DataGenerator(
            test_idx, y, batch_size=32, vec_len=vec_len, M=M,
            gains_I=gains, gains_Q=gains2,
            offsets_I=[0]*M, offsets_Q=[0]*M,
            skews_I=[0]*M, skews_Q=[0]*M,
            normalize=False, shuffle=True
        )
        score = model.evaluate(generator, verbose=0)
        trial_result['gain'] = score
        print(f"Score (GAIN only): {score}")

        # OFFSET only
        generator = DataGenerator(
            test_idx, y, batch_size=32, vec_len=vec_len, M=M,
            gains_I=[1]*M, gains_Q=[1]*M,
            offsets_I=offsets, offsets_Q=offsets2,
            skews_I=[0]*M, skews_Q=[0]*M,
            normalize=False, shuffle=True
        )
        score = model.evaluate(generator, verbose=0)
        trial_result['offset'] = score
        print(f"Score (OFFSET only): {score}")

        # SKEW only
        generator = DataGenerator(
            test_idx, y, batch_size=32, vec_len=vec_len, M=M,
            gains_I=[1]*M, gains_Q=[1]*M,
            offsets_I=[0]*M, offsets_Q=[0]*M,
            skews_I=skews, skews_Q=skews2,
            normalize=False, shuffle=True
        )
        score = model.evaluate(generator, verbose=0)
        trial_result['skew'] = score
        print(f"Score (SKEW only): {score}")

        # ALL mismatches
        generator = DataGenerator(
            test_idx, y, batch_size=32, vec_len=vec_len, M=M,
            gains_I=gains, gains_Q=gains2,
            offsets_I=offsets, offsets_Q=offsets2,
            skews_I=skews, skews_Q=skews2,
            normalize=False, shuffle=True
        )
        score = model.evaluate(generator, verbose=0)
        trial_result['all'] = score
        print(f"Score (ALL mismatches): {score}")

        # Store trial results
        results[f'M{M}'].append(trial_result)



# Print average scores across x trials
for M in [2, 4]:
    print(f"\n--- Average Scores for M = {M} ---")
    keys = ['gain', 'offset', 'skew', 'all']
    for k in keys:
        avg = np.mean([r[k][1] for r in results[f'M{M}']])  # [1] = accuracy
        print(f"{k.upper():<8}: {avg:.4f}")








def plot_confusion_matrix(cm, title='IQ Confusion matrix', fig='tester4.png', cmap=plt.cm.Blues, labels=[], fontsize=40):
    plt.figure()
    plt.xticks(fontsize=30)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=15)
    plt.yticks(tick_marks, labels, fontsize=15)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.savefig(fig, bbox_inches='tight')
    
    


'''
1) find all index positions where snr has specific value
2) keep only the ones that exist also in the test_index
3) use these values for evaluation
'''


def indices_in_range2(index_lst, snr_lst, x, y):
    snr_selection = [index for index, value in enumerate(snr_lst) if x <= value <= y]   # returns the index (position in list) of values in lst between x and y (including x and y) in the whole snr
    print('snr_selection: ', snr_selection[:10])
    print('index_lst: ', index_lst[:10])
    # Using List Comprehension
    common_indices = [x for x in snr_selection if x in index_lst]  #checks and keeps snr if the index also exists in the index list, here test_idx
    print('common_indices: ', common_indices[:10])
    #common_indices2 = [x for x in index_lst if x in snr_lst]
    #print('common_indices2: ', common_indices2[:10])
    return common_indices
    
    



'''
between -3 and 0
between 0-3
between 3-6
between 6-9
between 9-13
'''


# lower_bound 
# upper_bound 

acc = {}
def snr_conf(test_idx, y, snr, lb, ub):

    idx_snr = indices_in_range2(test_idx, snr, lb, ub)  

    
    selected_vectors = y[idx_snr]
    label_counts = np.sum(selected_vectors, axis=0)
    print('label_counts: ', label_counts)
    
    #print('idx_snr: ', idx_snr[:10])
    #print('y: ', y[:10])

    # estimate classes  note: shuffle for predict needs to be False otherwise it will missmatch with labels
    test_Y_hat = model.predict(DataGenerator(
            idx_snr, y, 1, vec_len=vec_len, M=M,
            gains_I=gains, gains_Q=gains2,
            offsets_I=offsets, offsets_Q=offsets2,
            skews_I=skews, skews_Q=skews2,
            normalize=False, shuffle=False
        ) , verbose=0 )    #  , verbose=1
    
    score = model.evaluate(DataGenerator(
            idx_snr, y, 1, vec_len=vec_len, M=M,
            gains_I=gains, gains_Q=gains2,
            offsets_I=offsets, offsets_Q=offsets2,
            skews_I=skews, skews_Q=skews2,
            normalize=False, shuffle=True
        ) , verbose=0 )   #  , verbose=1
    print('score: ', score)
    
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    
    for i in range(np.shape(idx_snr)[0]):
        j = list(y[idx_snr[i],:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
 
    #plot_confusion_matrix(confnorm, labels=classes, title="Dyadic Confusion Matrix (SNR=%d)"%(snr))
    plot_confusion_matrix(confnorm, labels=classes, title="IQ Confusion Matrix Function (SNR=%d-%d)"%(lb, ub), fig="Dyadic_pics_bench/IQ_Confusion_Matrix_Function_(SNR=%d-%d)_trials_50.png"%(lb, ub) )  
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    oa = cor / (cor+ncor)
    print ("Overall Accuracy: ", oa )
    
    print("Confusion Matrix:\n", conf)
    print("Correctly Classified Instances (cor):", cor)
    print("Total Instances (np.sum(conf)):", np.sum(conf))
    print("Misclassified Instances (ncor):", ncor)
    
    print(' ')
    
    return oa

    



oa_1 = snr_conf(test_idx, y, snr, -3, 0)
oa_2 = snr_conf(test_idx, y, snr, 0, 3)
oa_3 = snr_conf(test_idx, y, snr, 3, 6)
oa_4 = snr_conf(test_idx, y, snr, 6, 12)
#oa_5 = snr_conf(test_idx, y, snr, 9, 13)

    
  
#%%
# !!!
# Plot accuracy curve

snr_hist = [-1.5, 1.5, 4.5, 9.0]
acc_hist = [oa_1, oa_2, oa_3, oa_4]

print('acc_hist: ', acc_hist )

# orig 20 fontsize and 6 for linewidth
plt.figure()
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
#plt.plot(snr, list(map(lambda x: acc[x], snr_hist)), linewidth=9 )
plt.plot(snr_hist, acc_hist, linewidth=9 )
plt.xlabel("Signal to Noise Ratio", fontsize=30)
plt.ylabel("Classification Accuracy", fontsize=30)
plt.title("IQ Classification Accuracy ", fontsize=30)
plt.savefig('Dyadic_pics_bench/iq_snr_curve_trials_50.png', bbox_inches='tight')


print(' ')
print(' ')
print('end of code')



"""






