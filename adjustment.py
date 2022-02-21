# -*- coding: utf-8 -*-
"""
Created on 21/02/2022

@author: Yu-Chen Lin
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from scipy.io import wavfile
import tensorflow as tf
import pdb
import scipy.io
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Your GPU number, default = 0
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)
import time  
import numpy as np
import numpy.matlib
import random
random.seed(999)

from Call_back import *

def creatdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            #if filepath.split('/')[-2] != '-15db' and filepath.split('/')[-2] != '-10db':
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.     


def float16_quan(weights):
    weights_quan = []
    for layer_w in weights:
        w_flat = layer_w.flatten()
        w_flat = w_flat.tolist()
        w_after = []
        for w in w_flat:
            a = np.array(w, dtype=np.float32)
            b = a.astype(np.float16)
            c = b.astype(np.float32)
            w_after.append(c)
        w_after = np.array(w_after)
        w_after = np.reshape(w_after,layer_w.shape)
        weights_quan.append(w_after)
    return weights_quan



######################### Test_set #########################
'''
Test_Noisy_lists  = get_filepaths("/home/dwadelin/Corpus/TIMIT_SE/Test/Noisy/") #testing noisy set
Test_Clean_paths = "/home/dwadelin/Corpus/TIMIT_SE/Test/Clean/" # testing clean set 
'''  

MdNamePath='12_QFCN_TIMIT_epoch_100_mask_0' #the model path
with open(MdNamePath+'.json') as f:
    model = model_from_json(f.read());
        
model.load_weights(MdNamePath+'.hdf5');

mod_weights = model.get_weights()
new_weights = float16_quan(mod_weights)
model.set_weights(new_weights)
model.save_weights("new.hdf5")


pdb.set_trace()

'''
print(K.floatx())
print('testing...')
for path in Test_Noisy_lists: # Ex: /mnt/Nas/Corpus/TMHINT/Testing/Noisy/car_noise_idle_noise_60_mph/b4/1dB/TMHINT_12_10.wav
    S=path.split('/') 
    noise=S[-4]
    speaker=S[-3]
    dB=S[-2]    
    wave_name=S[-1]
    
    rate, noisy = wavfile.read(path)
    noisy=noisy.astype('float32')
    if len(noisy.shape)==2:
        noisy=(noisy[:,0]+noisy[:,1])/2
   
    noisy=noisy/np.max(abs(noisy))
    noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

    enhanced=np.squeeze(model.predict(noisy, verbose=0, batch_size=batch_size))
    enhanced=enhanced/np.max(abs(enhanced))
    enhanced=enhanced.astype('float32')
    creatdir(os.path.join("12_QFCN_TIMIT_epoch_100_mask_0_enhanced", noise, speaker, dB))
    librosa.output.write_wav(os.path.join("12_QFCN_TIMIT_epoch_100_mask_0_enhanced", noise, speaker, dB, wave_name), enhanced, 16000)
tEnd = time.time()
print "It cost %f sec" % (tEnd - tStart)

# plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print('drawing the training process...')
plt.figure(2)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('12_QFCN_TIMIT_epoch_100_mask_0.png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
'''
