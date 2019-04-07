import json
import random as rm
import numpy as np

dictionary={'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>':3, ',':4, '.':5, '!':6, '\"':7, '\u201c':8, '\u201d':9}
subdict={}
with open ('MLDS_hw2_1_data/training_label.json', 'r') as f:
    training_label = json.load(f)

training_label_dict={}
for d in training_label:
    training_label_dict[d['id']]=rm.choice(d['caption'])

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
    if subdict[w]>2:
        coun=len(dictionary)
        dictionary[w]=coun

dict_vec={}
for i, w in enumerate(dictionary):
    v=[]
    for j in range(len(dictionary)):
        if j==i:
            v.append(1)
        else:
            v.append(0)
    dict_vec[w]=v

with open('dict.json', 'w') as f:
    json.dump(dict_vec,f)

with open('training_name.json', 'w') as f:
    json.dump(training_label_dict,f)
sentence={}
longest=0
#print(training_label_dict)
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

        if '.' in w:
            a=news.index(w)
            news[a]=w[:w.index('.')]
            news.append('.')

        if '!' in w:
            a=news.index(w)
            news[a]=w[:w.index('!')]
            news.append('!')
    sentence[d]=news

with open('id_toword.json', 'w') as f:
    json.dump(sentence,f)