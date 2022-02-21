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
os.environ["CUDA_VISIBLE_DEVICES"]="1" #Your GPU number, default = 0
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
KTF.set_session(session)
import time  
import numpy as np
import numpy.matlib
import random
random.seed(999)

Num_traindata= 16200
epoch=300
batch_size=1
train_portion = 540
valid_portion = 30
## Total Number smaller than the training utterances in each case (Noise_type,SNR_Level) e.g., 540+30 <= 600


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

def data_generator(noisy_list, clean_path, shuffle = "False"):
    index=0
    while True:     
         random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')         
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2       
    
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1])
         clean=clean.astype('float')  
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2
         #print(noisy,clean)
         clean=clean/np.max(abs(clean))         
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))
         
         index += 1
         if index == len(noisy_list):
             index = 0
             if shuffle == "True":
                random.shuffle(noisy_list)
                      
         yield noisy, clean 

def valid_generator(noisy_list, clean_path, shuffle = "False"):
    index=0
    while True:
         random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2

         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1])
         clean=clean.astype('float')  
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2

         clean=clean/np.max(abs(clean))         
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))
         
         
         index += 1
         if index == len(noisy_list):
             index = 0
             if shuffle == "True":
                random.shuffle(noisy_list)
                       
         yield noisy, clean 

######################### Training data #########################
Train_Noisy_lists = get_filepaths("/mnt/Corpus/TIMIT_SE/Train/Noisy/")#training noisy set
Train_Clean_paths = "/mnt/Corpus/TIMIT_SE/Train/Clean/"

Train_lists = []
Valid_lists = []

for ind in range(30):
    random.shuffle(Train_Noisy_lists[ind*600:(ind+1)*600])
    train_temp = Train_Noisy_lists[ind*600:ind*600+train_portion]
    valid_temp = Train_Noisy_lists[600*(ind+1)-valid_portion:600*(ind+1)]
    Train_lists.extend(train_temp)
    Valid_lists.extend(valid_temp)

Num_Val_data = len(Valid_lists)
random.shuffle(Train_lists)

Train_lists=Train_lists[0:Num_traindata]      # Only use subset of training data

steps_per_epoch = (Num_traindata)//batch_size

######################### Test_set #########################
Test_Noisy_lists  = get_filepaths("/mnt/Corpus/TIMIT_SE/Test/Noisy/") #testing noisy set
Test_Clean_paths = "/mnt/Corpus/TIMIT_SE/Test/Clean/" # testing clean set 
#pdb.set_trace()
#Num_testdata=len()

     
start_time = time.time()

print ('model building...')

model = Sequential()


model.add(Convolution1D(30, 55, border_mode='same', input_shape=(None,1)))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(15, 35,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(1, 55,  border_mode='same'))
model.add(Activation('tanh'))

model.compile(loss='mse', optimizer='adam')
    
with open('12_layer_FCN_TIMIT.json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='12_layer_FCN_TIMIT.hdf5', verbose=1, save_best_only=True, mode='min')  
#pdb.set_trace()
'''
layer_void = model.layers
weights = layer_void[0].get_weights()
#print(weights[0][55//2])
#weights[0][55//2] = 0   

#for layer in model.layers:
    #print(layer.trainable)
#layer_void[0].trainable = False
#pdb.set_trace()
print(weights[0][(55//2)-9:(55//2)-3+9])
weights[0][(55//2)-9:(55//2)-3+9] = 0
print(weights[0][(55//2)-9:(55//2)-3+9])
layer_void[0].trainable = False
'''
print('training...')

g1 = data_generator(Train_lists, Train_Clean_paths, shuffle = "True")
g2 = valid_generator(Valid_lists, Train_Clean_paths, shuffle = "False")                					

print('g1 g2 generator')
#for e in range(epoch):
    #for dn in range(Num_traindata):
         #step = ["epoch:", e, ", process:", dn] 
         #print(step)
         #weights[0][(55//2)-6:(55//2)-3+6] = 0
hist=model.fit_generator(g1,    
                         samples_per_epoch=Num_traindata, 
                         nb_epoch=epoch, 
                         verbose=1,
                         validation_data=g2,
                         nb_val_samples=Num_Val_data,
                         max_q_size=10, 
                         nb_worker=1,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )                                   
#tf.reset_default_graph()
#print(weights[0][55//2])

tStart = time.time()

print('load model')
MdNamePath='12_layer_FCN_TIMIT' #the model path
with open(MdNamePath+'.json') as f:
    model = model_from_json(f.read());
        
model.load_weights(MdNamePath+'.hdf5');
model.summary()
pdb.set_trace()
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
    creatdir(os.path.join("12_layer_FCN_TIMIT_enhanced", noise, speaker, dB))
    librosa.output.write_wav(os.path.join("12_layer_FCN_TIMIT_enhanced", noise, speaker, dB, wave_name), enhanced, 16000)
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
plt.savefig('12_layer_FCN_TIMIT.png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

