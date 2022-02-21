# -*- coding: utf-8 -*-
"""
Created on 21/02/2022

@author: Yu-Chen Lin
"""

import keras
import random
import numpy as np
import bitstring
import struct

#def set_weight_file_name(w_name):
#    global save_weight_name
#    save_weight_name = w_name

######## quantize with floating point ########
def mask_bit_width(width):
    global bit_width
    bit_width = width

def set_hdf5_name(hdf5_name):
    global hdf5_file_name
    hdf5_file_name = hdf5_name

def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def quantization_floating_point(weights, bits_2_0):
    # bit_mask = '1'*(32-bits_2_0)+'0'*bits_2_0
    weights_quant = []

    if bits_2_0==0:
        return weights

    for layer_w in weights:
        flat_w = layer_w.flatten()
        flat_w = flat_w.tolist()
        after_mask_w = []
        for w in flat_w:
            bit_w = float_to_bin(w)
            
            if bit_w[-bits_2_0]=='1':
                after_mask_bit = bit_w[:-(bits_2_0+1)]+'1'+'0'*bits_2_0
            else:
                after_mask_bit = bit_w[:-bits_2_0]+'0'*bits_2_0

            after_mask_dec = bin_to_float(after_mask_bit)
            after_mask_w.append(after_mask_dec)
        after_mask_w = np.array(after_mask_w)
        after_mask_w = np.reshape(after_mask_w, layer_w.shape)
        weights_quant.append(after_mask_w)
    return weights_quant

weight_before_epoch = []
weight_after_epoch = []

class Mycallback_epoch(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        mod_weights = self.model.get_weights()
        weight_before_epoch.append(mod_weights)
        return

    def on_epoch_end(self, epoch, logs={}):
        mod_weights = self.model.get_weights()
        weight_after_epoch.append(mod_weights)
        q_weight = quantization_floating_point(mod_weights, bit_width)
        self.model.set_weights(q_weight)
        return

    def on_train_end(self, logs={}):        
        mod_weights = self.model.get_weights()
        q_weight = quantization_floating_point(mod_weights, bit_width)
        self.model.set_weights(q_weight)
        self.model.save_weights(hdf5_file_name)
        return

class Mycallback_none(keras.callbacks.Callback):
    def on_train_end(self, logs={}):
        self.model.save_weights(hdf5_file_name)
        return
