import matplotlib.pyplot as plt
import numpy as np

shuffle_ratio=0.1

for i in range(7):
    #load history
    test_acc=np.load('test_acc_MNIST-{}p.npy'.format(i))
    test_loss=np.load('test_loss_MNIST-{}p.npy'.format(i))
    train_acc=np.load('train_acc_MNIST-{}p.npy'.format(i))
    train_loss=np.load('train_loss_MNIST-{}p.npy'.format(i))

    plt.plot(test_acc)
    plt.plot(train_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy with {}%'.format(10+5*i))
    plt.legend(['test','train'], loc='lower right')
    plt.savefig('partial_random_acc{}.png'.format(i))
    plt.clf()

    plt.plot(test_loss,c='red')
    plt.plot(train_loss,c='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss with {}%'.format(10+5*i))
    plt.legend(['test','train'], loc='upper right')
    plt.savefig('partial_random_loss{}.png'.format(i))
    plt.clf()