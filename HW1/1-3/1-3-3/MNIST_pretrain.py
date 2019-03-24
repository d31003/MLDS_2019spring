import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from keras import backend as K
import h5py
from keras import optimizers

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
model.add(Conv2D(20, kernel_size=(3, 3), activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros', padding='SAME',input_shape=input_shape))
model.add(Flatten())
model.add(Dense(256, activation='softmax',  kernel_initializer='lecun_normal', bias_initializer='zeros'))
#model.add(Dense(256, activation='softmax',  kernel_initializer='lecun_normal', bias_initializer='zeros'))
model.add(Dense(10, activation='softmax',  kernel_initializer='lecun_normal', bias_initializer='zeros'))

adam=optimizers.Adam(lr=1e-2)
#print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


Result=model.fit(train_array_normalize, train_label_array,epochs=10,batch_size=1024,verbose=1,shuffle=True,validation_data=(test_array_normalize, test_label_array))
weights=model.get_weights()
weights=np.array(weights)
np.save("weights_lr2.npy", weights)
