import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py

rowsImage, colsImage = 28, 28

epoch = 15
batch = 200
shuffle_ratio = 0.1

for num in range(7):#10~40%
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

    #shuffle  all=60000
    for i in range( int(60000*(shuffle_ratio+num*0.05)) ):
        train_label[i]=(train_label[i]+np.random.randint(1,10,1))%10
    
    #one-hot encoding
    train_label_array=to_categorical(train_label)
    test_label_array=to_categorical(test_label)


    #model
    model = Sequential()
    model.add(Conv2D(15, kernel_size=(3, 3), activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros', padding='SAME',input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(10, activation='softmax',  kernel_initializer='lecun_normal', bias_initializer='zeros'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


    Result=model.fit(train_array_normalize,train_label_array,epochs=epoch,batch_size=batch,verbose=1,shuffle=True,validation_data=(test_array_normalize, test_label_array))
    model.save('MNIST-{}p.h5'.format(num))
    np.save("train_loss_MNIST-{}p.npy".format(num),Result.history['loss'])
    np.save("train_acc_MNIST-{}p.npy".format(num),Result.history['acc'])
    np.save("test_loss_MNIST-{}p.npy".format(num),Result.history['val_loss'])
    np.save("test_acc_MNIST-{}p.npy".format(num),Result.history['val_acc'])
    print("\n\n\nMODEL {} DONE!\n\n\n".format(num))