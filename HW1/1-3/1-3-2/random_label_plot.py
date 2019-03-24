import matplotlib.pyplot as plt
import numpy as np

x=[]
test_acc=[]
train_acc=[]
test_loss=[]
train_loss=[]

x.append(2354360)
x.append(4708560)
x.append(7062760)
x.append(9416960)
x.append(11771160)
x.append(14125360)
x.append(16479560)
x.append(18833760)
x.append(21187960)
x.append(23542160)


#load history
for i in range(1,11):
    a=np.load('test_acc_MNIST-{}.npy'.format(i))
    test_acc.append(a[-1])
    a=np.load('test_loss_MNIST-{}.npy'.format(i))
    test_loss.append(a[-1])
    a=np.load('train_acc_MNIST-{}.npy'.format(i))
    train_acc.append(a[-1])
    a=np.load('train_loss_MNIST-{}.npy'.format(i))
    train_loss.append(a[-1])

plt.scatter(x,test_acc,c='red')
plt.scatter(x,train_acc,c='blue')
plt.xlabel('parameters')
plt.ylabel('accuracy')
plt.legend(['test','train'],loc='center right')
plt.savefig('random_acc.png')
plt.clf()

plt.scatter(x,test_loss,c='red')
plt.scatter(x,train_loss,c='blue')
plt.xlabel('parameters')
plt.ylabel('loss')
plt.legend(['test','train'],loc='upper right')
plt.savefig('random_loss.png')