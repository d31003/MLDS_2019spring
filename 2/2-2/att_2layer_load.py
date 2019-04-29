import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import LSTM, Input, Softmax, TimeDistributed, dot, concatenate, Embedding, Reshape, Concatenate
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

EPOCH = 1
BATCH_SIZE=500
sample_round= 10000
dict_min=600
unk=1
longest=8
sentence_size=12
sample_num=3000
latent_dim = 1024

with open ('dictionary{}.json'.format(dict_min), 'r') as f:
    dictionary = json.load(f)

#with open ('id_toword.json', 'r') as f:# movie name: caption sentence
#    training_label_dict = json.load(f)
#with open ('dict{}.json'.format(dict_min), 'r') as f:
#    one_hot = json.load(f)

input_e=np.load('input_e_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size))
input_d=np.load('input_d_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size))
gt=np.load('gt_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size))
pad=np.zeros((sample_num, sentence_size, len(dictionary)))
pad[:,:,0]=1


num_encoder_tokens = len(dictionary)
num_decoder_tokens = len(dictionary)


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_sequences=True)
encoder_outputs = encoder(encoder_inputs) 
encoder_last = encoder_outputs[:,-1,:]
pad_inputs = Input(shape=(None, num_encoder_tokens))
encoder1_inputs = concatenate([pad_inputs, encoder_outputs],axis=-1)
encoder1 = LSTM(latent_dim, return_sequences=True)
encoder1_outputs = encoder1([encoder1_inputs]) 
encoder1_last = encoder1_outputs[:,-1,:]
# Set up the decoder, using `encoder_states` as initial state.

#decoder_inputs = Input(shape=(None,num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences=True)
decoder_outputs=decoder(pad_inputs, initial_state=[encoder_last, encoder_last])

decoder1_inputs = Input(shape=(None,num_decoder_tokens))#(None, num_decoder_tokens)
decoder1_inputs_c = concatenate([decoder1_inputs, decoder_outputs],axis=-1)
decoder1= LSTM(latent_dim, return_sequences=True)
decoder1_outputs= decoder1(decoder1_inputs_c, initial_state=[encoder1_last, encoder1_last])#, d_h, d_c

attention0 = dot([decoder1_outputs, encoder1_outputs], axes=[2,2])
attention = Activation('softmax')(attention0)

context = dot([attention, encoder1_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder1_outputs])

# Has another weight + tanh layer as described in equation (5) of the paper
output = TimeDistributed(Dense(latent_dim, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
output = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))(output)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs,decoder1_inputs, pad_inputs], output)
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
    #    model=load_model('model/att_sample_{}_unk_d{}_epoch{}_dim{}_{}.h5'.format(sample_num,dict_min,EPOCH,latent_dim))
    
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


    model.fit([train_data, gt_oh, pad],train_label,epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, shuffle=True)
    if T%1000==0:
        model.save('model/att_s2vt_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}_{}.h5'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim, T))
    del train_data
    del train_label
    del gt_oh
    gc.collect()


model.save('model/att_s2vt_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}.h5'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim))


