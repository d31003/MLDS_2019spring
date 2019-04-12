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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCH = 15
sample_num=20
dict_min=5
unk_max=2

with open ('dictionary{}.json'.format(dict_min), 'r') as f:
    dictionary = json.load(f)

#with open ('id_toword.json', 'r') as f:# movie name: caption sentence
#    training_label_dict = json.load(f)
with open ('dict{}.json'.format(dict_min), 'r') as f:
    one_hot = json.load(f)

with open ('MLDS_hw2_1_data/training_label.json', 'r') as f:
    training_label = json.load(f)

num_encoder_tokens = 4096
num_decoder_tokens = len(dictionary)
latent_dim = 256

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens)) #(None, 4096)
encoder = LSTM(latent_dim, return_sequences=True)
encoder_outputs= encoder(encoder_inputs) #, state_h, state_c 
# We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]
#print(encoder_outputs.shape)
encoder_last = encoder_outputs[:,-1,:]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,1))#(None, num_decoder_tokens)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder = Embedding(len(dictionary), latent_dim)(decoder_inputs)#, input_length=(35)
decoder=Reshape((35,latent_dim))(decoder)
decoder_lstm= LSTM(latent_dim, return_sequences=True)
decoder_outputs= decoder_lstm(decoder, initial_state=[encoder_last, encoder_last])#, d_h, d_c

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

out = [] # first decoder input
for i in range(1450):
    out.append(np.array(dictionary["<BOS>"]).reshape(1,1))
out=np.array(out)

training_label_dict={}
for d in training_label:
    training_label_dict[d['id']]=rm.choice(d['caption'])

train_data=[]
for d in training_label_dict:
    a=np.load("./MLDS_hw2_1_data/training_data/feat/{}.npy".format(d))
    a=a.reshape(80,4096)
    a=np.concatenate((a,np.zeros((1,4096))),axis=0)#PAD
    #a=np.concatenate((a,np.ones((1,4096))),axis=0)#BOS
    train_data.append(a)
train_data=np.array(train_data)



for T in range(sample_num):
    print('round', T)
    rechoice=[]
    training_label_dict={}
    for d in training_label:
        training_label_dict[d['id']]=rm.choice(d['caption'])

    sentence={}
    for d in training_label_dict:
        s=training_label_dict[d].split(' ')
        #print(s)
        news=[]
        for w in s:
            news.append(w)
            if ',' in w:
                a=news.index(w)
                news[a]=w[:w.index(',')]
                news.append(',')
                del a

            if '.' in w:
                a=news.index(w)
                news[a]=w[:w.index('.')]
                news.append('.')
                del a

            if '!' in w:
                a=news.index(w)
                news[a]=w[:w.index('!')]
                news.append('!')
                del a
        del s
        sentence[d]=news
        del news

    

    

    for d in sentence:
        sentence[d]+=['<EOS>']
        l=len(sentence[d])
        for i in range(35-l):
            sentence[d]+=['<PAD>']
        del l

    train_label=[]
    train_label1=[]
    for d in sentence:
        a=[]
        b=[]
        unk=0
        for w in sentence[d]:
            if w in dictionary:
                a.append([dictionary[w]])
                b.append(one_hot[w])
            else:
                a.append([dictionary["<UNK>"]])
                b.append(one_hot["<UNK>"])
                unk+=1
                if unk==unk_max:
                    rechoice.append(d)
        train_label.append(a)
        train_label1.append(b)

        del unk
        del a
        del b
    #train_label=np.array(train_label)

    
    for t in range(10):
        
        for d in training_label:
            if d['id'] in rechoice:
                #print(d['id'])
                training_label_dict[d['id']]=rm.choice(d['caption'])

        sentence={}
        for d in training_label_dict:
            if d in rechoice:
                #print(d)
                s=training_label_dict[d].split(' ')
                #print(s)
                news=[]
                for w in s:
                    news.append(w)
                    if ',' in w:
                        a=news.index(w)
                        news[a]=w[:w.index(',')]
                        news.append(',')
                        del a

                    if '.' in w:
                        a=news.index(w)
                        news[a]=w[:w.index('.')]
                        news.append('.')
                        del a

                    if '!' in w:
                        a=news.index(w)
                        news[a]=w[:w.index('!')]
                        news.append('!')
                        del a
                del s
                sentence[d]=news
                del news

        

        

        for d in sentence:
            sentence[d]+=['<EOS>']
            l=len(sentence[d])
            for i in range(35-l):
                sentence[d]+=['<PAD>']
            del l
        del rechoice
        rechoice=[]
        #train_label=[]
        for i, d in enumerate(sentence):
            a=[]
            b=[]
            unk=0
            for w in sentence[d]:
                if w in dictionary:
                    a.append([dictionary[w]])
                    b.append(one_hot[w])
                else:
                    a.append([dictionary["<UNK>"]])
                    b.append(one_hot["<UNK>"])
                    unk+=1
                    if unk==unk_max:
                        rechoice.append(d)
            train_label[i]=a
            train_label1[i]=b
            del a
            del b

        if rechoice==[]:
            #print('break')
            break
    train_label=np.array(train_label)
    train_label1=np.array(train_label1)

    print(train_label.shape, out.shape)

    
    gt=np.concatenate((out,train_label[:,:34,:]),axis=1)
    model.fit([train_data,gt],train_label1,epochs=EPOCH, batch_size=10, verbose=1, shuffle=True)

    #model.save_weights('att_weight_sample_3_epoch100.h5')
    del sentence
    del training_label_dict
    #del train_data
    del gt
    #del news
    del rechoice
    del train_label
    del train_label1
    gc.collect()


model.save('att_sample_{}_unk_epoch{}.h5'.format(sample_num,EPOCH))



### PREDICT (INFERNECE)

with open ('MLDS_hw2_1_data/testing_label.json', 'r') as f:
    testing_label = json.load(f)
ID=[]
for d in testing_label:
    ID.append(d['id'])

test_data=[]
for i in ID:
    a=np.load("./MLDS_hw2_1_data/testing_data/feat/{}.npy".format(i))
    a=a.reshape(80,4096)
    a=np.concatenate((a,np.zeros((1,4096))),axis=0)#PAD
    #a=np.concatenate((a,np.ones((1,4096))),axis=0)#BOS
    test_data.append(a)
test_data=np.array(test_data) #100, 81, 4096
num_decoder_tokens = len(dictionary)
"""out=[]
for i in range(len(ID)):
    out.append(np.array(dictionary["<BOS>"]).reshape(1,1))
out=np.array(out)"""

with open ('vec2word{}.json'.format(dict_min), 'r') as f:
    vec2word = json.load(f)

def decode_sequence(input_seq):
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # out (100,1,2497)
    test_decoder_input=[]
    for i in range(100):
        test_decoder_input.append([])
        z=np.zeros((34,1))
        a=np.array(dictionary["<BOS>"]).reshape(1,1)
        test_decoder_input[i].append( np.concatenate((a,z), axis=0) )
    test_decoder_input=np.array(test_decoder_input)
    test_decoder_input=test_decoder_input.reshape((100,35,1))
    print("shape", test_decoder_input.shape)

    decoded_sentence = []

    for i in range(100):
        decoded_sentence.append('')
    
    for i in range(35):
        output_tokens = model.predict([input_seq, test_decoder_input])

        for k in range(100):
            
            #Sample a token
            sample=output_tokens[k][i]
            sample_num=np.argmax(sample)
            single_word=vec2word[str(sample_num)]
            decoded_sentence[k] += (' ' + single_word)

            #update states!
            if i != 34:
                test_decoder_input[k][i+1][0]=sample_num

    for i in range(100):
        #print(decoded_sentence)
        tem = decoded_sentence[i].split("<PAD>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split("<EOS>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split(".",1)
        decoded_sentence[i]=tem[0]
    return decoded_sentence

d_s=decode_sequence(test_data)

answer=''
for i in range(len(ID)):
    answer+= (ID[i] + ',' + d_s[i] + '\n')

f=open('att_sample_{}_unk_epoch{}.txt'.format(sample_num,EPOCH),'w')
f.write(answer)
