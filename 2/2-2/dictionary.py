import json
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dict_min=600
unk=1
longest=8
sentence_size=12
correlation = 0.5

dictionary={'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>':3}#, ',':4, '.':5, '!':6, '?':7, '。':8, '～':9, '-':10, ';':11
subdict={}
question=open('sel_conversation/answer.txt','r').readlines()
answer=open('sel_conversation/question.txt','r').readlines()

question_sentences=[]
answer_sentences=[]

lq=0
la=0

for i in range(len(question)):
    question_sentences.append(question[i].split())
    answer_sentences.append(answer[i].split())
    for w in question_sentences[i]:
        if w in subdict:
            subdict[w]+=1
        else:
            subdict[w]=0
    for w in answer_sentences[i]:
        if w in subdict:
            subdict[w]+=1
        else:
            subdict[w]=0
    if len(question_sentences[i])>lq:
        lq=len(question_sentences[i])
    if len(answer_sentences[i])>la:
        la=len(question_sentences[i])
print(lq, la)

for w in subdict:
    if subdict[w]>dict_min-1:
        coun=len(dictionary)
        dictionary[w]=coun
with open('dictionary{}.json'.format(dict_min), 'w') as f:
    json.dump(dictionary,f)
'''
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
'''
vec2word={}
for w in dictionary:
    vec2word[int(dictionary[w])]=w
with open('vec2word{}.json'.format(dict_min), 'w') as f:
    json.dump(vec2word,f)
print(len(dictionary))

input_e=[]
input_d=[]
gt=[]

for i in range(len(question_sentences)):
    if len(question_sentences[i])<=longest and len(answer_sentences[i])<=longest:
        bufq=[]
        uq=0
        for w in question_sentences[i]:
            if w in dictionary:
                bufq.append(dictionary[w])
            else:
                uq+=1
                if uq>unk:
                    break
                bufq.append(dictionary['<UNK>'])
        if uq<=unk:
            for j in range(sentence_size-len(bufq)):
                bufq.append(dictionary['<PAD>'])
            bufa=[]
            ua=0
            for w in answer_sentences[i]:
                if w in dictionary:
                    bufa.append(dictionary[w])
                else:
                    ua+=1
                    if ua>unk:
                        break
                    bufa.append(dictionary['<UNK>'])
                
            if ua<=unk:
                bufg=[dictionary['<BOS>']]+bufa.copy()
                bufa.append(dictionary['<EOS>'])
                for j in range(sentence_size-len(bufa)):
                    bufa.append(dictionary['<PAD>'])
                for j in range(sentence_size-len(bufg)):
                    bufg.append(dictionary['<PAD>'])
                input_e.append(bufq)
                input_d.append(bufa)
                gt.append(bufg)
input_e=np.array(input_e)
input_d=np.array(input_d)
gt=np.array(gt)

#np.save('input_e_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size), input_e)
#np.save('input_d_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size), input_d)
#np.save('gt_d{}_unk{}_len_{}_size{}.npy'.format(dict_min, unk, longest, sentence_size), gt)
np.save('input_opp_e.npy', input_e)
np.save('input_opp_d.npy', input_d)
np.save('gt_opp.npy', gt)

print(input_e.shape, input_d.shape, gt.shape)
