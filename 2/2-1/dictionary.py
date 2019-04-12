import json
import numpy as np

dict_min=10

dictionary={'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>':3, ',':4, '.':5, '!':6, '\"':7, '\u201c':8, '\u201d':9}
subdict={}
with open ('MLDS_hw2_1_data/training_label.json', 'r') as f:
    training_label = json.load(f)

AllSentences=[]
for d in training_label:
    AllSentences+=d['caption']

for i,d in enumerate(AllSentences):

    words=AllSentences[i].split(' ')
    for w in words:
        if ',' in w:
            w=w[:w.index(',')]

        if '.' in w:
            w=w[:w.index('.')]

        if '!' in w:
            w=w[:w.index('!')]

        if w in subdict:
            subdict[w]+=1
        else:
            subdict[w]=0

for w in subdict:
    if subdict[w]>dict_min-1:
        coun=len(dictionary)
        dictionary[w]=coun
with open('dictionary{}.json'.format(dict_min), 'w') as f:
    json.dump(dictionary,f)

#one-hot
dict_vec={}
for i, w in enumerate(dictionary):
    v=[]
    for j in range(len(dictionary)):
        if j==i:
            v.append(1)
        else:
            v.append(0)
    dict_vec[w]=v

with open('dict{}.json'.format(dict_min), 'w') as f:
    json.dump(dict_vec,f)

vec2word={}
for w in dictionary:
    vec2word[int(dictionary[w])]=w
with open('vec2word{}.json'.format(dict_min), 'w') as f:
    json.dump(vec2word,f)
print(len(dictionary))