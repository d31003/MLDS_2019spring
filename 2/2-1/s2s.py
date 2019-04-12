import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py
import json
import tensorflow as tf


with open ('dictionary.json', 'r') as f:
    dictionary = json.load(f)

with open ('id_toword.json', 'r') as f:# movie name: caption sentence
    training_label_dict = json.load(f)
with open ('dict.json', 'r') as f:
    one_hot = json.load(f)


train_data=[]
'''z=np.zeros((1,64,64,1))

for d in training_label_dict:
    a=np.load("./MLDS_hw2_1_data/training_data/feat/{}.npy".format(d))
    a=a.reshape(80,64,64,1)
    a=np.concatenate((a,z),axis=0)
    train_data.append(a)
'''
for d in training_label_dict:
    a=np.load("./MLDS_hw2_1_data/training_data/feat/{}.npy".format(d))
    a=a.reshape(80,4096)
    a=np.concatenate((a,np.zeros((1,4096))),axis=0)#PAD
    #a=np.concatenate((a,np.ones((1,4096))),axis=0)#BOS
    train_data.append(a)
train_data=np.array(train_data)

for d in training_label_dict:
    training_label_dict[d]+=['<EOS>']
    l=len(training_label_dict[d])
    for i in range(35-l):
        training_label_dict[d]+=['<PAD>']

train_label=[]
for d in training_label_dict:
    a=[]
    for w in training_label_dict[d]:
        if w in one_hot:
            a.append(one_hot[w])
        else:
            a.append(one_hot["<UNK>"])
    train_label.append(a)
train_label=np.array(train_label)

#train_label=[] #ideal outputs


#80, 4096  1450 movies
# configure
num_encoder_tokens = 4096
num_decoder_tokens = len(dictionary)
latent_dim = 512

out = [] # first decoder input
for i in range(1450):
    out.append(np.array(one_hot["<BOS>"]).reshape(1,num_decoder_tokens))
out=np.array(out)


gt=np.concatenate((out,train_label[:,0:34,:]),axis=1)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# print the model
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')#,metrics=['accuracy']
Result=model.fit([train_data,gt],train_label,epochs=300,batch_size=10,verbose=1,shuffle=True)#,validation_data=(test_array_normalize, test_label_array)

model.save('s2s_basic_dim512.h5')


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
test_data=np.array(test_data)
num_decoder_tokens = len(dictionary)
out=[]
for i in range(len(ID)):
    out.append(np.array(one_hot["<BOS>"]).reshape(1,num_decoder_tokens))
out=np.array(out)

with open ('vec2word.json', 'r') as f:
    vec2word = json.load(f)

# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)
# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 定義解碼器 LSTM 模型
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    ini_decoder_input = out #(100,1,2497)
    decoded_sentence = []
    for i in range(100):
        decoded_sentence.append('')
    
    for i in range(35):
        output_tokens, h, c = decoder_model.predict([ini_decoder_input] + states_value)
        ini_decoder_input=[]
        for k in range(100):
            
            #Sample a token
            sample=output_tokens[k][0]
            sample_num=np.argmax(sample)
            single_word=vec2word[str(sample_num)]
            decoded_sentence[k] += (' ' + single_word)

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, num_decoder_tokens))
            target_seq[0, sample_num] = 1.0
            ini_decoder_input.append(target_seq)
        # Update states
        states_value = [h, c]
        ini_decoder_input=np.array(ini_decoder_input)

    for i in range(100):
        tem = decoded_sentence[i].split("<PAD>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split("<EOS>",1)
        decoded_sentence[i]=tem[0]
        tem = decoded_sentence[i].split(".",1)
        decoded_sentence[i]=tem[0]
    return decoded_sentence

d_s=decode_sequence(test_data)
#print(d_s)
answer=''
for i in range(len(ID)):
    answer+= (ID[i] + ',' + d_s[i] + '\n')


f=open('s2s_basic_dim512.txt','w')
f.write(answer)
