"""
Created on Tue Aug 25 04:01:41 PM 2020
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import hdf5storage
import numpy as np
from keras.utils import plot_model
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.constraints import unit_norm
import time
import sys
import scipy.io
import argparse
import librosa
import pdb
import random

random.seed(999)
pathmodel="/Data/user_mhb/se/DDAE3/Model_OPENSLR_DDAE3_6L_fw4_200ep_2kN_lr3e5_drpOT15"
MxEpoch = 200
batch_size = 1
FRAMESIZE = 512
OVERLAP = 256
FRAMEWIDTH = 4
FBIN = FRAMESIZE//2+1  # =257
input_dim = FBIN*(FRAMEWIDTH*2+1)

FFTSIZE = 512
RATE = 16000
neuron= 2048
dropOut= 0.15

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    

def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    return FileList;


######################### Training data #########################
Train_Noisy_paths = get_filenames("/Data/user_mhb/se/list/train_noisy_orig.txt")
#Train_Noisy_paths=Train_Noisy_paths[0:100]
Train_Noisy_wavename=[]
for path in Train_Noisy_paths:
    S=path.split('/')[-1]
    #pdb.set_trace()
    Train_Noisy_wavename.append(S)
    #pdb.set_trace()
    
Train_Clean_paths = get_filenames("/Data/user_mhb/se/list/train_clean_orig.txt")
#Train_Clean_paths=Train_Clean_paths[0:100]
Train_Clean_wavename=[]
for path in Train_Clean_paths:
    S=path.split('/')[-1]
    Train_Clean_wavename.append(S)
   
# data_shuffle
Num_traindata=len(Train_Noisy_paths)
permute = list(range(Num_traindata))
random.shuffle(permute)

Train_Noisy_paths=shuffle_list(Train_Noisy_paths,permute)
Train_Noisy_wavename=shuffle_list(Train_Noisy_wavename,permute)

######################### Test_set #########################
Test_Noisy_paths = get_filenames("/Data/user_mhb/se/list/test_noisy_orig.txt")
#Test_Noisy_paths=Test_Noisy_paths[0:50]
Test_Noisy_wavename=[]
for path in Test_Noisy_paths:
    S=path.split('/')[-1]
    Test_Noisy_wavename.append(S)
    
Test_Clean_paths = get_filenames("/Data/user_mhb/se/list/test_clean_orig.txt")  
#Test_Clean_paths=Test_Clean_paths[0:50]
Test_Clean_wavename=[]
Test_Clean_paths_prune=[]
for path in Test_Clean_paths:
    S=path.split('/')[-1]
    if  S[-3:]=='wav':
        Test_Clean_wavename.append(S)
        Test_Clean_paths_prune.append(path)
                
Num_testdata=len(Test_Noisy_paths)    

######################### Training Stage ########################
data = train_data_extractor(Train_Noisy_paths, Train_Noisy_wavename, Train_Clean_paths, Train_Clean_wavename)
data_eval = val_data_generator(Test_Noisy_paths, Test_Noisy_wavename, Test_Clean_paths_prune, Test_Clean_wavename)

model = Sequential()

# Create multi-layer model and set parameter in each layer
model.add(Dense(neuron, activation='relu', input_shape=(input_dim,)))
model.add(Dropout(dropOut))
model.add(Dense(neuron, activation='relu'))
model.add(Dropout(dropOut))
model.add(Dense(neuron, activation='relu'))
model.add(Dropout(dropOut))
model.add(Dense(neuron, activation='relu'))
model.add(Dropout(dropOut))
model.add(Dense(neuron, activation='relu'))
model.add(Dropout(dropOut))
#model.add(Dense(neuron, activation='relu'))
#model.add(Dropout(0.35))
#model.add(Dense(neuron, activation='relu'))
#model.add(Dropout(0.35))
#model.add(Dense(257, activation='linear')) 
#model.add(Dense(257, activation='tanh'))
model.add(Dense(257))
model.summary()

adam=Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)   
model.compile(loss='mse', optimizer=adam)
with open(pathmodel+".json", "w") as f:
          f.write(model.to_json())
#pdb.set_trace()
checkpointer = ModelCheckpoint(filepath=pathmodel+".hdf5", monitor="loss", verbose=1, save_best_only=True, mode='min')
hist=model.fit_generator(data, steps_per_epoch=Num_traindata, epochs=MxEpoch,  verbose=1, 
                            validation_data=data_eval, validation_steps=Num_testdata,
                            max_queue_size=10, workers=1,callbacks=[checkpointer])

