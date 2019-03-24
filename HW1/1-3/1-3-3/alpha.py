import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py
import matplotlib.pyplot as plt

rowsImage, colsImage = 28, 28

#load data
(train_image,train_label),(test_image,test_label)=mnist.load_data()
if K.image_data_format() == 'channels_first':
    train_image = train_image.reshape((train_image.shape[0], 1, rowsImage, colsImage))
    test_image = test_image.reshape(test_image.shape[0], 1, rowsImage, colsImage)
    input_shape = (1, rowsImage, colsImage)
else:
    train_image = train_image.reshape((train_image.shape[0], rowsImage, colsImage, 1))
    test_image = test_image.reshape(test_image.shape[0], rowsImage, colsImage, 1)
    input_shape = (rowsImage, colsImage, 1)

#normalize
train_array_normalize = train_image.astype('float32')/255.0
test_array_normalize = test_image.astype('float32')/255.0

#one-hot encoding
train_label_array=to_categorical(train_label)
test_label_array=to_categorical(test_label)

#model  118000para
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3), activation='selu', padding='SAME',input_shape=input_shape))#, kernel_initializer='lecun_normal', bias_initializer='zeros'
model.add(Flatten())
model.add(Dense(256, activation='softmax'))#,  kernel_initializer='lecun_normal', bias_initializer='zeros'))
#model.add(Dense(256, activation='softmax',  kernel_initializer='lecun_normal', bias_initializer='zeros'))
model.add(Dense(10, activation='softmax'))#,  kernel_initializer='lecun_normal', bias_initializer='zeros'))

b64=np.load("weights_batch64.npy")
b1024=np.load("weights_batch1024.npy")
alpha=np.arange(-1,2,0.1)
trl=[]
tra=[]
tel=[]
tea=[]
#print('alpha=',alpha)
#print('b64=',b64)
#alpha=[0,1]
for a in alpha:
    theta=b64*a+b1024*(1-a)
    model.set_weights(theta)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    trainloss, trainacc = model.evaluate(train_array_normalize, train_label_array)
    testloss, testacc = model.evaluate(test_array_normalize, test_label_array)
    #print(trainloss,trainacc,testloss, testacc)
    trl.append(trainloss)
    tra.append(trainacc)
    tel.append(testloss)
    tea.append(testacc)
plt.plot(alpha, trl, label='train')
plt.plot(alpha, tel, label='test')
plt.legend(['train','test'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('cross_entropy')
plt.savefig('batchsize_loss.png')
plt.clf()

plt.plot(alpha, tra, label='train_accuracy')
plt.plot(alpha, tea, label='test_accuracy')
plt.legend(['train','test'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('batchsize_acc.png')
plt.clf()

lr2=np.load("weights_lr1e-2.npy")
lr3=np.load("weights_lr1e-3.npy")
alpha=np.arange(-1,2,0.1)
trl=[]
tra=[]
tel=[]
tea=[]
for a in alpha:
    theta=a*lr3+(1-a)*lr2
    model.set_weights(theta)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    trainloss, trainacc = model.evaluate(train_array_normalize, train_label_array)
    testloss, testacc = model.evaluate(test_array_normalize, test_label_array)
    trl.append(trainloss)
    tra.append(trainacc)
    tel.append(testloss)
    tea.append(testacc)
plt.plot(alpha, trl, label='train')
plt.plot(alpha, tel, label='test')
plt.legend(['train','test'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('cross_entropy')
plt.savefig('lr_loss.png')
plt.clf()

plt.plot(alpha, tra, label='train_accuracy')
plt.plot(alpha, tea, label='test_accuracy')
plt.legend(['train','test'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.savefig('lr_acc.png')

