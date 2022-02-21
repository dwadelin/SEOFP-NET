# -*- coding: utf-8 -*-
"""
Created on 21/02/2022

@author: Yu-Chen Lin
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import scipy.io
import soundfile as sf
import os
import time  
import numpy as np
import numpy.matlib
import random
import sys

from scipy.io import wavfile
from pystoi.stoi import stoi
from pypesq import pesq

def creatdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filepaths(directory, dtype='.wav'):
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
            if filepath.split('/')[-1][-4:] == dtype:
                file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

######################### Evaluation_set #########################
work = "./score/"+sys.argv[1]

if sys.argv[2] == 'ref':
    Test_list_a = get_filepaths("/mnt/Corpus/TIMIT_SE/Test/Noisy/BabyCry", '.wav')
    Test_list_b = get_filepaths("/mnt/Corpus/TIMIT_SE/Test/Noisy/Engine", '.wav')
    Test_list_c = get_filepaths("/mnt/Corpus/TIMIT_SE/Test/Noisy/White", '.wav')
else:
    Test_list_a = get_filepaths("./wav/"+sys.argv[1]+"/Noisy/BabyCry", '.wav')
    Test_list_b = get_filepaths("./wav/"+sys.argv[1]+"/Noisy/Engine", '.wav')
    Test_list_c = get_filepaths("./wav/"+sys.argv[1]+"/Noisy/White", '.wav')
#pdb.set_trace()
#Test_lists = []
#Test_lists.extend(Test_list_a)
#Test_lists.extend(Test_list_b)
#Test_lists.extend(Test_list_c)

print("Start Test_Babycry Evaluation ...")

a=0
b=0
c=0
d=0
e=0
f=0

pesq_a=0
pesq_b=0
pesq_c=0
pesq_d=0
pesq_e=0
pesq_f=0

stoi_a=0
stoi_b=0
stoi_c=0
stoi_d=0
stoi_e=0
stoi_f=0
#n=0

for path in Test_list_a:
    #n=n+1
    #print(n) 
    N = path.split('/')

    #rate, ref = wavfile.read('/mnt/md2/user_khhung/vad/Aurora2/wavfile/'+N[-4]+'/clean/'+N[-1])
    #rate, deg = wavfile.read(path)
    #ref, rate = sf.read('/mnt/md2/user_khhung/vad/Aurora2/wavfile/'+N[-4]+'/clean/'+N[-1])
    #deg, rate = sf.read(path)
    rate, ref = wavfile.read('/mnt/Corpus/TIMIT_SE/Test/Clean/'+N[-1])
    ref=ref.astype('float')
    if len(ref.shape)==2:
        ref=(ref[:,0]+ref[:,1])/2
    rate, deg = wavfile.read(path)
    deg=deg.astype('float')
    if len(deg.shape)==2:
        deg=(deg[:,0]+ref[:,1])/2
    #print(ref.shape, deg.shape)
    
    #if len(ref) < 12000:
    #    continue
    #   ref=np.concatenate((ref, np.zeros(40000-len(ref))))
    #   deg=np.concatenate((deg, np.zeros(40000-len(deg))))
    #print(ref.shape, deg.shape)
    #pdb.set_trace()
    ref = ref.astype('float32')/np.max(abs(ref.astype('float32')))
    deg = deg.astype('float32')/np.max(abs(deg.astype('float32')))
    #pdb.set_trace()
    if len(ref) != len(deg):
        ref = ref[0:np.min(len(ref),len(deg))]
        deg = deg[0:np.min(len(ref),len(deg))]
    #print(ref, deg)
    #pdb.set_trace()
    #print(path,len(ref))
    pesq_temp = pesq(ref, deg, rate)
    stoi_temp = stoi(ref, deg, rate, extended=False)
    #print(pesq_temp, stoi_temp)
    if np.isnan(pesq_temp) or np.isnan(stoi_temp):
        continue       
        
    creatdir(os.path.join(work))
    #pdb.set_trace()
    if N[-2]=='-15dB':
        a=a+1
        #pdb.set_trace()
        pesq_a=pesq_a+pesq_temp
        stoi_a=stoi_a+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-15dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-12dB':
        b=b+1
        pesq_b=pesq_b+pesq_temp
        stoi_b=stoi_b+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-6dB':
        c=c+1
        pesq_c=pesq_c+pesq_temp
        stoi_c=stoi_c+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='0dB':
        d=d+1
        pesq_d=pesq_d+pesq_temp
        stoi_d=stoi_d+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/0dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='6dB':
        e=e+1
        pesq_e=pesq_e+pesq_temp
        stoi_e=stoi_e+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='12dB':
        f=f+1
        #print(f)
        pesq_f=pesq_f+pesq_temp
        stoi_f=stoi_f+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()
#pdb.set_trace()
cnt = a+b+c+d+e+f
print(a,b,c,d,e,f,cnt)
head = str(["Count: ", cnt])
fp = open("./"+work+"/"+"BabyCry_Average"+".txt", "a")
fp.write(head)
fp.write("\n")
line = str(["PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f])
fp.write(line)
fp.write("\n")
line = str(["PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e])
fp.write(line)
fp.write("\n")
line = str(["PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d])
fp.write(line)
fp.write("\n")
line = str(["PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c])
fp.write(line)
fp.write("\n")
line = str(["PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b])
fp.write(line)
fp.write("\n")
line = str(["PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a])
fp.write(line)
fp.write("\n")

#pdb.set_trace()
print("PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f)
print("PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e)
print("PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d)
print("PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c)
print("PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b)
print("PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a)

print("Start Test_Engine Evaluation ...")

a=0
b=0
c=0
d=0
e=0
f=0

pesq_a=0
pesq_b=0
pesq_c=0
pesq_d=0
pesq_e=0
pesq_f=0

stoi_a=0
stoi_b=0
stoi_c=0
stoi_d=0
stoi_e=0
stoi_f=0

for path in Test_list_b:

    N = path.split('/')

    #rate, ref = wavfile.read('/mnt/md2/user_khhung/vad/Aurora2/wavfile/'+N[-4]+'/clean/'+N[-1])
    #rate, deg = wavfile.read(p)
    rate, ref = wavfile.read('/mntCorpus/TIMIT_SE/Test/Clean/'+N[-1])
    ref=ref.astype('float')
    if len(ref.shape)==2:
        ref=(ref[:,0]+ref[:,1])/2
    rate, deg = wavfile.read(path)
    deg=deg.astype('float')
    if len(deg.shape)==2:
        deg=(deg[:,0]+ref[:,1])/2
    #print(ref.shape, deg.shape)
    #if len(ref) < 12000:
    #   continue
    #ref=np.concatenate((ref, ref))
    #deg=np.concatenate((deg, deg))

    ref = ref.astype('float32')/np.max(abs(ref.astype('float32')))
    deg = deg.astype('float32')/np.max(abs(deg.astype('float32')))

    if len(ref) != len(deg):
        ref = ref[0:np.min(len(ref),len(deg))]
        deg = deg[0:np.min(len(ref),len(deg))]
    #print(ref, deg)
    pesq_temp = pesq(ref, deg, rate)
    stoi_temp = stoi(ref, deg, rate, extended=False)

    if np.isnan(pesq_temp) or np.isnan(stoi_temp):
        continue
    
    creatdir(os.path.join(work))

    if N[-2]=='-15dB':
        a=a+1
        #pdb.set_trace()
        pesq_a=pesq_a+pesq_temp
        stoi_a=stoi_a+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-15dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-12dB':
        b=b+1
        pesq_b=pesq_b+pesq_temp
        stoi_b=stoi_b+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-6dB':
        c=c+1
        pesq_c=pesq_c+pesq_temp
        stoi_c=stoi_c+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='0dB':
        d=d+1
        pesq_d=pesq_d+pesq_temp
        stoi_d=stoi_d+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/0dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='6dB':
        e=e+1
        pesq_e=pesq_e+pesq_temp
        stoi_e=stoi_e+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='12dB':
        f=f+1
        #print(f)
        pesq_f=pesq_f+pesq_temp
        stoi_f=stoi_f+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()
#pdb.set_trace()
cnt = a+b+c+d+e+f
print(a,b,c,d,e,f,cnt)
head = str(["Count: ", cnt])
fp = open("./"+work+"/"+"Engine_Average"+".txt", "a")
fp.write(head)
fp.write("\n")
line = str(["PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f])
fp.write(line)
fp.write("\n")
line = str(["PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e])
fp.write(line)
fp.write("\n")
line = str(["PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d])
fp.write(line)
fp.write("\n")
line = str(["PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c])
fp.write(line)
fp.write("\n")
line = str(["PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b])
fp.write(line)
fp.write("\n")
line = str(["PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a])
fp.write(line)
fp.write("\n")

#pdb.set_trace()
print("PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f)
print("PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e)
print("PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d)
print("PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c)
print("PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b)
print("PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a)

print("Start Test_White Evaluation ...")

a=0
b=0
c=0
d=0
e=0
f=0

pesq_a=0
pesq_b=0
pesq_c=0
pesq_d=0
pesq_e=0
pesq_f=0

stoi_a=0
stoi_b=0
stoi_c=0
stoi_d=0
stoi_e=0
stoi_f=0

for path in Test_list_c:

    N = path.split('/')

    #rate, ref = wavfile.read('/mnt/md2/user_khhung/vad/Aurora2/wavfile/'+N[-4]+'/clean/'+N[-1])
    #rate, deg = wavfile.read(p)

    rate, ref = wavfile.read('/mnt/Corpus/TIMIT_SE/Test/Clean/'+N[-1])
    ref=ref.astype('float')
    if len(ref.shape)==2:
        ref=(ref[:,0]+ref[:,1])/2
    rate, deg = wavfile.read(path)
    deg=deg.astype('float')
    if len(deg.shape)==2:
        deg=(deg[:,0]+ref[:,1])/2
    #print(ref.shape, deg.shape)
    #if len(ref) < 12000:
    #   continue
    #ref=np.concatenate((ref, ref))
    #deg=np.concatenate((deg, deg))

    ref = ref.astype('float32')/np.max(abs(ref.astype('float32')))
    deg = deg.astype('float32')/np.max(abs(deg.astype('float32')))

    if len(ref) != len(deg):
        ref = ref[0:np.min(len(ref),len(deg))]
        deg = deg[0:np.min(len(ref),len(deg))]
    #print(ref, deg)
    pesq_temp = pesq(ref, deg, rate)
    stoi_temp = stoi(ref, deg, rate, extended=False)

    if np.isnan(pesq_temp) or np.isnan(stoi_temp):
        continue
    
    creatdir(os.path.join(work, N[-4]))

    if N[-2]=='-15dB':
        a=a+1
        #pdb.set_trace()
        pesq_a=pesq_a+pesq_temp
        stoi_a=stoi_a+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-15dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-12dB':
        b=b+1
        pesq_b=pesq_b+pesq_temp
        stoi_b=stoi_b+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='-6dB':
        c=c+1
        pesq_c=pesq_c+pesq_temp
        stoi_c=stoi_c+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/-6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='0dB':
        d=d+1
        pesq_d=pesq_d+pesq_temp
        stoi_d=stoi_d+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/0dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='6dB':
        e=e+1
        pesq_e=pesq_e+pesq_temp
        stoi_e=stoi_e+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/6dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()

    elif N[-2]=='12dB':
        f=f+1
        #print(f)
        pesq_f=pesq_f+pesq_temp
        stoi_f=stoi_f+stoi_temp
        line = str(["PESQ:  ", pesq_temp, "STOI:  ", stoi_temp])
        fp = open("./"+work+"/12dB.txt", "a")
        fp.write(line)
        fp.write("\n")
        fp.close()
#pdb.set_trace()
cnt = a+b+c+d+e+f
print(a,b,c,d,e,f,cnt)
head = str(["Count: ", cnt])
fp = open("./"+work+"/"+"White_Average"+".txt", "a")
fp.write(head)
fp.write("\n")
line = str(["PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f])
fp.write(line)
fp.write("\n")
line = str(["PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e])
fp.write(line)
fp.write("\n")
line = str(["PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d])
fp.write(line)
fp.write("\n")
line = str(["PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c])
fp.write(line)
fp.write("\n")
line = str(["PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b])
fp.write(line)
fp.write("\n")
line = str(["PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a])
fp.write(line)
fp.write("\n")

#pdb.set_trace()
print("PESQ 12dB:", pesq_f/f, "STOI 12dB:", stoi_f/f)
print("PESQ 6dB:", pesq_e/e, "STOI 6dB:", stoi_e/e)
print("PESQ 0dB:", pesq_d/d, "STOI 0dB:", stoi_d/d)
print("PESQ -6dB:", pesq_c/c, "STOI -6dB:", stoi_c/c)
print("PESQ -12dB:", pesq_b/b, "STOI -12dB:", stoi_b/b)
print("PESQ -15dB:", pesq_a/a, "STOI -15dB:", stoi_a/a)

print("End Evaluation ...")
    


