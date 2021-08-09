# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 01:07:35 2017
@author: Jason
Edited by Md Mahbub E Noor Thurs Sep 03 02:11:30 PM 2020, here i did with 5 neighbouring frame based feature extraction, where it was used only current single frame earlier
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.activations import softmax
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input
from keras.constraints import max_norm
from keras.utils import plot_model
from keras import regularizers
import scipy.io
import librosa
import os
import time  
import math
import numpy as np
import numpy.matlib
import random
import pdb
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
#import theano
#import theano.tensor as T

random.seed(999)
name = "OpenSlr_CNN2D_mhbV2.10_fw4"
pathmodel = "/Data/user_mhb/se_openslr2/Model_" + name
epoch = 200
batch_size = 1
FRAMESIZE = 512
OVERLAP = 256
FRAMEWIDTH = 4
FBIN = FRAMESIZE//2+1  # =257
input_dim = FBIN*(FRAMEWIDTH*2+1)

FFTSIZE = 512
RATE = 16000
#dropOut= 0.15

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new    

def get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    #pdb.set_trace()
    return FileList;
   
######################### Training data #########################
Train_Noisy_paths = get_filenames("/Data/user_mhb/se_openslr2/list/compressed_tr10k.txt")
#Train_Noisy_paths=Train_Noisy_paths[0:100]
Train_Noisy_wavename=[]
for path in Train_Noisy_paths:
    S=path.split('/')[-1]
    #pdb.set_trace()
    Train_Noisy_wavename.append(S)
    #pdb.set_trace()
    
Train_Clean_paths = get_filenames("/Data/user_mhb/se_openslr2/list/clean_tr10k.txt")
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
######################### Test_set #############################
Test_Noisy_paths = get_filenames("/Data/user_mhb/se_openslr2/list/compressed_valid.txt")
#Test_Noisy_paths=Test_Noisy_paths[0:138]
Test_Noisy_wavename=[]
for path in Test_Noisy_paths:
    S=path.split('/')[-1]
    Test_Noisy_wavename.append(S)
    
Test_Clean_paths = get_filenames("/Data/user_mhb/se_openslr2/list/clean_valid.txt")  
#Test_Clean_paths=Test_Clean_paths[0:138]
Test_Clean_wavename=[]
Test_Clean_paths_prune=[]
for path in Test_Clean_paths:
    S=path.split('/')[-1]
    if  S[-3:]=='wav':
        Test_Clean_wavename.append(S)
        Test_Clean_paths_prune.append(path)
                
Num_testdata=len(Test_Noisy_paths)    
######################### Training Stage ########################           
start_time = time.time()

print('model building...')

_input = Input(shape=(None, input_dim))

re_input = keras.layers.core.Reshape((-1, input_dim, 1), input_shape=(-1, input_dim))(_input)
        
# CNN
conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
conv1= (BatchNormalization())(conv1)
conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
conv1= (BatchNormalization())(conv1)
conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
conv1= (BatchNormalization())(conv1)
out_shape1 = math.floor((input_dim-3+2)/3)+1  #output_size= ((input_size-filter_size+2*padding_size)/stride_size)+1
        
conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
conv2= (BatchNormalization())(conv2)
conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
conv2= (BatchNormalization())(conv2)
conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
conv2= (BatchNormalization())(conv2)
out_shape2 = math.floor((out_shape1-3+2)/3)+1
        
conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
conv3= (BatchNormalization())(conv3)
conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
conv3= (BatchNormalization())(conv3)
conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
conv3= (BatchNormalization())(conv3)
out_shape3 = math.floor((out_shape2-3+2)/3)+1
        
conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
conv4= (BatchNormalization())(conv4)
conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
conv4= (BatchNormalization())(conv4)
conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
conv4= (BatchNormalization())(conv4)
out_shape4 = math.floor((out_shape3-3+2)/3)+1

conv5 = (Conv2D(256, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
conv5= (BatchNormalization())(conv5)
conv5 = (Conv2D(256, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv5)
conv5= (BatchNormalization())(conv5)
conv5 = (Conv2D(256, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv5)
conv5= (BatchNormalization())(conv5)
out_shape5 = math.floor((out_shape4-3+2)/3)+1

re_shape = keras.layers.core.Reshape((-1, out_shape5*256), input_shape=(-1, out_shape5, 256))(conv5)
dropout_layer1=(Dropout(0.20))(re_shape)
final_layer=TimeDistributed(Dense(257))(dropout_layer1)
model = Model(outputs=final_layer, inputs=_input)
model.summary()

adam=Adam(lr=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)

plot_model(model, to_file = name + '.png', show_shapes=True)
checkpointer = ModelCheckpoint(filepath=pathmodel+".hdf5", monitor="loss", verbose=1, save_best_only=True, mode='min')  
print('training...')

g1 = train_data_generator(Train_Noisy_paths, Train_Noisy_wavename, Train_Clean_paths, Train_Clean_wavename)
g2 = val_data_generator  (Test_Noisy_paths, Test_Noisy_wavename, Test_Clean_paths, Test_Clean_wavename)


hist=model.fit_generator(g1,	steps_per_epoch=Num_traindata, 
                            epochs=epoch, 
                            verbose=1,
                            validation_data=g2,
                            validation_steps=Num_testdata,
                            max_queue_size=1, 
                            workers=1,
                            callbacks=[checkpointer]
                            )
with open(pathmodel + ".json",'w') as f:    # save the model
    f.write(model.to_json()) 
    

