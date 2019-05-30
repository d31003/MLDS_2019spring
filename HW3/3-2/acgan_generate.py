from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import numpy as np
import sys
import cv2

# py cuda epoch
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

T = int(sys.argv[2])

np.random.seed(126)

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 120
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator

        # Build the generator
        self.generator = self.build_generator()

        

    def build_generator(self):

        generator=Sequential()
        generator.add(Dense(50*16*16, input_dim=self.latent_dim))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Reshape((16,16,50)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Conv2D(3, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Flatten())
        generator.add(Dense(64*64*3, activation='tanh'))
        generator.add(Reshape((64,64,3)))

        generator.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = generator(model_input)

        return Model([noise, label], img)


def save_imgs(generator):
    import matplotlib.pyplot as plt

    # 5 tags at the same time
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    label = [99,99,99,99,99, 66,66,66,66,66, 48,48,48,48,48, 4,4,4,4,4, 74,74,74,74,74]
    label = np.array(label).reshape(-1,1)

    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict([noise, label])
    gen_imgs = (gen_imgs+1)/2
    gen_imgs.astype(float)

    # bgr -> rgb
    image_list=[]
    for i in range(r*c):
        #image_list.append(cv2.cvtColor(gen_imgs[i], cv2.COLOR_BGR2RGB))
        image_list.append(gen_imgs[i,:,:,::-1])
    image_list = np.array(image_list)
    gen_imgs = image_list

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output.png")
    plt.close()


generator = CGAN().generator
generator.load_weights('model/acgan_g_{}.h5'.format(T))


save_imgs(generator)
