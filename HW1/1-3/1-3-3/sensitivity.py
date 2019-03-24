from keras import backend as K
from keras.models import load_model
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam


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

model50 = load_model('MNIST_batch50.h5')
model100 = load_model('MNIST_batch100.h5')
model500 = load_model('MNIST_batch500.h5')
model1000 = load_model('MNIST_batch1000.h5')
model2000 = load_model('MNIST_batch2000.h5')


result50=np.load("result_b50.npy")
result100=np.load("result_b100.npy")
result500=np.load("result_b500.npy")
result1000=np.load("result_b1000.npy")
result2000=np.load("result_b2000.npy")

R=[result50,result100,result500,result1000,result2000]
R=np.array(R)
R=R.transpose()

test_array_normalize_=test_array_normalize+1e-5
#####
model50.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

testloss, testacc = model50.evaluate(test_array_normalize, test_label_array)
testloss_, testacc_ = model50.evaluate(test_array_normalize_, test_label_array)

grad50=(testloss-testloss_)/1e-5
#print(grad50)
gradnorm50=abs(grad50)
#gradnorm50=np.linalg.norm(grad50,ord='fro')
#####
#####
model100.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

testloss, testacc = model100.evaluate(test_array_normalize, test_label_array)
testloss_, testacc_ = model100.evaluate(test_array_normalize_, test_label_array)

grad100=(testloss-testloss_)/1e-5
gradnorm100=abs(grad100)
#gradnorm100=np.linalg.norm(grad100,ord='fro')
#####
#####
model500.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

testloss, testacc = model500.evaluate(test_array_normalize, test_label_array)
testloss_, testacc_ = model500.evaluate(test_array_normalize_, test_label_array)

grad500=(testloss-testloss_)/1e-5
gradnorm500=abs(grad500)
#gradnorm500=np.linalg.norm(grad500,ord='fro')
#####
#####
model1000.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

testloss, testacc = model1000.evaluate(test_array_normalize, test_label_array)
testloss_, testacc_ = model1000.evaluate(test_array_normalize_, test_label_array)

grad1000=(testloss-testloss_)/1e-5
gradnorm1000=abs(grad1000)
#gradnorm1000=np.linalg.norm(grad1000,ord='fro')
#####
#####
model2000.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

testloss, testacc = model2000.evaluate(test_array_normalize, test_label_array)
testloss_, testacc_ = model2000.evaluate(test_array_normalize_, test_label_array)

grad2000=(testloss-testloss_)/1e-5
gradnorm2000=abs(grad2000)
#gradnorm2000=np.linalg.norm(grad2000,ord='fro')
#####
gradnorm=[gradnorm50,gradnorm100,gradnorm500,gradnorm1000,gradnorm2000]
x=[50,100,500,1000,2000]
fig=plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x,R[0],label='train')
plt.plot(x,R[2],label='test')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper left')
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
plt.plot(x,gradnorm,label='sensitivity', color='red')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel('sensitivity')

plt.legend(['sensitivity'],loc='upper right')

plt.xlabel('batch_size')

plt.savefig("sensitivity_loss.png")
plt.clf()

fig=plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x,R[1],label='train')
plt.plot(x,R[3],label='test')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper left')
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)

plt.plot(x,gradnorm,label='sensitivity', color='red')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel('sensitivity')
plt.legend(['sensitivity'],loc='upper right')
plt.xlabel('batch_size')

plt.savefig("sensitivity_acc.png")
plt.clf()

