import numpy as np
import cv2
import json
'''
IMGs=[]
for i in range(36740):
    img=cv2.imread('./extra_data/images/{}.jpg'.format(i))
    #print(img.shape)  (96,96,3)
    img=cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
    #print(img.shape)
    IMGs.append(img)
IMGs=np.array(IMGs)
np.save('real_images.npy', IMGs)

tags=[]

hair_color={}
eyes_color={}
l=open('./extra_data/tags.csv').readlines()
for s in l:
    s=s.split(',')[1]
    s=s.split()
    h=s[0]
    e=s[2]
    x=[]
    if h in hair_color:
        x.append(hair_color[h])
    else:
        hair_color[h]=len(hair_color)
        x.append(hair_color[h])
    if e in eyes_color:
        x.append(eyes_color[e])
    else:
        eyes_color[e]=len(eyes_color)
        x.append(eyes_color[e])
    tags.append(x)

tags_v=[]
for t in tags:
    h_v=np.zeros(len(hair_color))
    e_v=np.zeros(len(eyes_color))
    h_v[t[0]]=1
    e_v[t[1]]=1
    tags_v.append(np.concatenate((h_v, e_v)))
tags_v=np.array(tags_v)
print(tags_v.shape)
print(tags_v[0])
np.save('tags.npy', tags_v)

with open('hair_color.json', 'w') as f:
    json.dump(hair_color,f)

with open('eyes_color.json', 'w') as f:
    json.dump(eyes_color,f)
'''
hair_eyes={}
tags_v=[]

l=open('./extra_data/tags.csv').readlines()
for s in l:
    #v=np.zeros(120)
    s=s.split(',')[1]
    if s not in hair_eyes:
        hair_eyes[s]=len(hair_eyes)
        #v[hair_eyes[s]]=1
    tags_v.append(hair_eyes[s])

tags_v=np.array(tags_v)
print(tags_v.shape)
print(tags_v[0])
np.save('tags.npy', tags_v)

with open('hair_eyes.json', 'w') as f:
    json.dump(hair_eyes,f)