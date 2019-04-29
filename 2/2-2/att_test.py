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
correlation=0.5

with open ('dictionary{}.json'.format(dict_min), 'r') as f:
    dictionary = json.load(f)


num_encoder_tokens = len(dictionary)
num_decoder_tokens = len(dictionary)


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens)) #(None, 4096)
encoder = LSTM(latent_dim, return_sequences=True)
encoder_outputs= encoder(encoder_inputs) #, state_h, state_c 
# We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]
#print(encoder_outputs.shape)
encoder_last = encoder_outputs[:,-1,:]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,num_decoder_tokens))#(None, num_decoder_tokens)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
#decoder = Embedding(len(dictionary), latent_dim)(decoder_inputs)#, input_length=(35)
#decoder=Reshape((35,latent_dim))(decoder)
decoder= LSTM(latent_dim, return_sequences=True)
decoder_outputs= decoder(decoder_inputs, initial_state=[encoder_last, encoder_last])#, d_h, d_c

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
#print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')



### PREDICT (INFERNECE)
test=open('mlds_hw2_2_data/test_input.txt','r').readlines()
test_data=[]
#ts=''
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


with open ('vec2word{}.json'.format(dict_min), 'r') as f:
    vec2word = json.load(f)


#model=load_model('model/att_sr{}_sn{}_unk{}_d{}_len{}_size{}_epoch{}_dim{}_{}.h5'.format(sample_round, sample_num, unk, dict_min, longest, sentence_size, EPOCH,latent_dim, T))
model=load_model('model/att_opp.h5')
def decode_sequence(input_seq):
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # out (100,1,2497)
    #test_decoder_input=[]
    test_decoder_input_oh=[]
    '''for i in range(len(test_data)):
        test_decoder_input.append([])
        z=np.zeros((sentence_size-1,1))
        a=np.array(dictionary["<BOS>"]).reshape(1,1)
        test_decoder_input[i].append( np.concatenate((a,z), axis=0) )
    '''
    for i in range(len(test_data)):
        test_decoder_input_oh.append([])
        z=np.zeros((sentence_size-1,len(dictionary)))
        a=np.zeros((1,len(dictionary)))
        a[0][1]=1
        test_decoder_input_oh[i].append( np.concatenate((a,z), axis=0) )

    #test_decoder_input=np.array(test_decoder_input)
    #test_decoder_input=test_decoder_input.reshape((len(test_data),sentence_size,1))
    test_decoder_input_oh=np.array(test_decoder_input_oh)
    test_decoder_input_oh=test_decoder_input_oh.reshape((len(test_data),sentence_size,len(dictionary)))
            

    decoded_sentence = []
    for k in range(len(test_data)):
        decoded_sentence.append('')
    
    
    for i in range(sentence_size):
        output_tokens = model.predict([input_seq, test_decoder_input_oh])

        for k in range(len(test_data)):
            
            #Sample a token
            
            sample=output_tokens[k][i]
            sample_Num=np.argmax(sample)
            single_word=vec2word[str(sample_Num)]
            decoded_sentence[k] += (single_word)

            #update states!
            if i < sentence_size-1:
                #test_decoder_input[k][i+1][0]=sample_Num
                a=np.zeros(len(dictionary))
                a[sample_Num]=1
                test_decoder_input_oh[k][i+1]=a

    for i in range(len(test_data)):
        #print(decoded_sentence)
        tem = decoded_sentence[i].split("<PAD>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split("<EOS>",1)
        decoded_sentence[i]=tem[0]
        #tem = decoded_sentence[i].split(".",1)
        #decoded_sentence[i]=tem[0]
        
    return decoded_sentence

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

f=open('txt/att_opp.txt'.format(correlation),'w')
f.write(answer)
