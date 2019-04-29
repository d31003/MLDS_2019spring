import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import LSTM, Input, Softmax, TimeDistributed, dot, concatenate, Embedding, Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py
import json
import tensorflow as tf
import random as rm
import os
import gc
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

EPOCH = 1
BATCH_SIZE=300
sample_round= 10000
dict_min=600
unk=1
longest=8
sentence_size=12
sample_num=3000
latent_dim = 1024
#correlation = 0.5

with open ('dictionary{}.json'.format(dict_min), 'r') as f:
    dictionary = json.load(f)

input_e=np.load('input_opp_e.npy')#'input_e_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size))
input_d=np.load('input_opp_d.npy')
gt=np.load('gt_opp.npy')


num_encoder_tokens = len(dictionary)
num_decoder_tokens = len(dictionary)


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_sequences=True)
encoder_outputs= encoder(encoder_inputs) 
encoder_last = encoder_outputs[:,-1,:]
# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None,num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences=True)
decoder_outputs=decoder(decoder_inputs, initial_state=[encoder_last, encoder_last])


attention0 = dot([decoder_outputs, encoder_outputs], axes=[2,2])
attention = Activation('softmax')(attention0)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

# Has another weight + tanh layer as described in equation (5) of the paper
output = TimeDistributed(Dense(latent_dim, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
output = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))(output)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], output)
# print the model
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')


#########
#model=load_model('model/att_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}_3440.h5'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim))
##########

for T in range(sample_round):#sample_round 
    print('round', T)
    train_data=[]
    train_label=[]
    gt_oh=[]
    #if T==0:
        #model=load_model('model/att_correlation_0.5_8000.h5')
    
    #if T==1:
    #    break
    
    for i in range(sample_num):
        x=rm.randint(0,len(input_d)-1)
        
        bufq=[]
        bufa=[]
        bufg=[]
        for w in input_e[x]:
            z=np.zeros(len(dictionary))
            z[w]=1
            bufq.append(z)
            del z
            #gc.collect()
        for w in input_d[x]:
            z=np.zeros(len(dictionary))
            z[w]=1
            bufa.append(z)
            del z
            #gc.collect()
        for w in gt[x]:
            z=np.zeros(len(dictionary))
            z[w]=1
            bufg.append(z)
            del z
            #gc.collect()
        train_data.append(bufq)
        train_label.append(bufa)
        gt_oh.append(bufg)
        del bufq
        del bufa
        del bufg
        #gc.collect()
    train_data=np.array(train_data)
    train_label=np.array(train_label)
    gt_oh=np.array(gt_oh)
    #print(train_data[-1], train_label[-1], gt_oh[-1])
    


    model.fit([train_data,gt_oh],train_label,epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, shuffle=True)
    if T%1000==0:
        model.save('model/att_opp_{}.h5'.format(T))
    del train_data
    del train_label
    del gt_oh
    gc.collect()


model.save('model/att_opp.h5')
