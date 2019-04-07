import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Input, Softmax, TimeDistributed, dot, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py
import json
import tensorflow as tf

EPOCH = 300

#reading dictionaries
with open ('dictionary.json', 'r') as f:
    dictionary = json.load(f)

with open ('id_toword.json', 'r') as f:# movie name: caption sentence
    training_label_dict = json.load(f)
with open ('dict.json', 'r') as f:
    one_hot = json.load(f)


train_data=[]

for d in training_label_dict:
    a=np.load("./MLDS_hw2_1_data/training_data/feat/{}.npy".format(d))
    a=a.reshape(80,4096)
    a=np.concatenate((a,np.zeros((1,4096))),axis=0)#PAD
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

#feature(80, 4096)  1450 movies
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
encoder_inputs = Input(shape=(None, num_encoder_tokens)) #(None, 4096)
encoder = LSTM(latent_dim, return_sequences=True)
encoder_outputs= encoder(encoder_inputs) #, state_h, state_c 
# We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]

encoder_last = encoder_outputs[:,-1,:]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm= LSTM(latent_dim, return_sequences=True)
decoder_outputs= decoder_lstm(decoder_inputs, initial_state=[encoder_last, encoder_last])#, d_h, d_c

attention0 = dot([decoder_outputs, encoder_outputs], axes=[2,2])
attention = Activation('softmax')(attention0)

context = dot([attention, encoder_outputs], axes=[2,1])
decoder_combined_context = concatenate([context, decoder_outputs])

# Has another weight + tanh layer
output = TimeDistributed(Dense(latent_dim, activation="tanh"))(decoder_combined_context)
output = TimeDistributed(Dense(num_decoder_tokens, activation="softmax"))(output)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], output)
# print the model
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')
Result=model.fit([train_data,gt],train_label,epochs=EPOCH, batch_size=10, verbose=1, shuffle=True)

model.save('s2s-att_epoch{}.h5'.format(EPOCH))



### PREDICT (INFERENCE)

#reading testing features and movies' IDs
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
    test_data.append(a)
test_data=np.array(test_data) #shape:100, 81, 4096
num_decoder_tokens = len(dictionary)
out=[]
for i in range(len(ID)):
    out.append(np.array(one_hot["<BOS>"]).reshape(1,num_decoder_tokens))
out=np.array(out)

with open ('vec2word.json', 'r') as f:
    vec2word = json.load(f)

def decode_sequence(input_seq):
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # out shape (100,1,2497)
    test_decoder_input=[]
    for i in range(100):
        test_decoder_input.append([])
        z=np.zeros((34,num_decoder_tokens))
        a=np.array(one_hot["<BOS>"]).reshape(1,num_decoder_tokens)
        test_decoder_input[i].append( np.concatenate((a,z), axis=0) )
    test_decoder_input=np.array(test_decoder_input)
    test_decoder_input=test_decoder_input.reshape((100,35,2497))

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
                test_decoder_input[k][i+1][sample_num]=1.0

    for i in range(100):
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

f=open('s2s-att_epoch{}.txt'.format(EPOCH),'w')
f.write(answer)
