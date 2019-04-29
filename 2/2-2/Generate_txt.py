import numpy as np
import sys

cor_min = float(sys.argv[1])

question=open('./../../sel_conversation/question.txt','r').readlines()
answer=open('./../../sel_conversation/answer.txt','r').readlines()

score = np.load('./../../sel_conversation/cor.npy')
print("Original Data size:", len(score))
fq = open('./../../sel_conversation/train_input_{}.txt'.format(cor_min), 'w')
fa = open('./../../sel_conversation/train_output_{}.txt'.format(cor_min), 'w')

a = ''
q = ''

count = 0

for i in range(len(score)):
    if score[i] >= cor_min:
        q += question[i]
        a += answer[i]
        count += 1

print("Improved Data size:", count)
fq.write(q)
fa.write(a)