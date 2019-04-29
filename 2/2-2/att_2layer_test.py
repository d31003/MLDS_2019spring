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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

EPOCH = 1
sample_round= 10000
dict_min=600
unk=1
longest=8
sentence_size=12
sample_num=3000
latent_dim = 1024

with open ('dictionary{}.json'.format(dict_min), 'r') as f:
    dictionary = json.load(f)


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
model = Model([encoder_inputs, decoder1_inputs, pad_inputs], output)


### PREDICT (INFERNECE)
test=open('mlds_hw2_2_data/test_input.txt','r').readlines()
test_data=[]

for s in test:
    l=s.split()

    for i in range(sentence_size-len(l)):
        l.append('<PAD>')
    buf=[]
    for w in l:
        z=np.zeros(len(dictionary))
        if w in dictionary:
            z[dictionary[w]]=1
        else:
            z[dictionary['<UNK>']]=1
        buf.append(z)

    test_data.append(buf)
test_data=np.array(test_data)

pad=np.zeros((len(test_data), sentence_size, len(dictionary)))
pad[:,:,0]=1

with open ('vec2word{}.json'.format(dict_min), 'r') as f:
    vec2word = json.load(f)

def decode_sequence(input_seq):
        
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # out (100,1,2497)
    #test_decoder_input=[]
    test_decoder_input_oh=[]

    for i in range(len(test_data)):
        test_decoder_input_oh.append([])
        z=np.zeros((sentence_size-1,len(dictionary)))
        a=np.zeros((1,len(dictionary)))
        a[0][1]=1
        test_decoder_input_oh[i].append( np.concatenate((a,z), axis=0) )

    test_decoder_input_oh=np.array(test_decoder_input_oh)
    test_decoder_input_oh=test_decoder_input_oh.reshape((len(test_data),sentence_size,len(dictionary)))    

    decoded_sentence = []
    for k in range(len(test_data)):
        decoded_sentence.append('')
    
    
    for i in range(sentence_size):
        output_tokens = model.predict([input_seq, test_decoder_input_oh, pad])

        for k in range(len(test_data)):
            
            #Sample a token
            sample=output_tokens[k][i]
            sample_Num=np.argmax(sample)
            single_word=vec2word[str(sample_Num)]
            decoded_sentence[k] += (single_word)

            #update states!
            if i < sentence_size-1:
                a=np.zeros(len(dictionary))
                a[sample_Num]=1
                test_decoder_input_oh[k][i+1]=a

    for i in range(len(test_data)):
        print(decoded_sentence[i])
        tem = decoded_sentence[i].split("<PAD>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split("<EOS>",1)
        decoded_sentence[i]=tem[0]

        
    return decoded_sentence


model=load_model('model/att_s2vt_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}.h5'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim))
d_s=decode_sequence(test_data)

# parsing <UNK>
answer=''
for i in range(len(test_data)):
    if d_s[i]!='<UNK>':
        tem = d_s[i].split('<UNK>')
    else:
        tem = [d_s[i]]
    for j in range(len(tem)):
        answer+=tem[j]
    answer+='\n'

f=open('txt/att_s2vt_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}.txt'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim),'w')
f.write(answer)