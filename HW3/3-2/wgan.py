from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import sys
import os
import numpy as np
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

d_iter=3

def wloss(y_true, y_pred):
    V = y_true*y_pred
    return K.mean(V)

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 120
        self.latent_dim = 100

        self.LossFunction = wloss
        
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.0002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.LossFunction,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.LossFunction,
            optimizer=optimizer)

    def build_generator(self):

        generator=Sequential()
        generator.add(Dense(128*16*16, input_dim=self.latent_dim))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Reshape((16,16,128)))
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

    def build_discriminator(self):

        discriminator=Sequential()
        discriminator.add(Dense(80*16*16, input_dim=np.prod(self.img_shape)))
        discriminator.add(LeakyReLU())
        discriminator.add(Reshape((16,16,80)))
        discriminator.add(Conv2D(32, kernel_size=4))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(64, kernel_size=4, padding='same'))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(128, kernel_size=4))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(256, kernel_size=4))
        discriminator.add(LeakyReLU())
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        #model_input = multiply([img, label_embedding])

        validity = discriminator(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128):

        

        IMGs=np.load('real_images.npy')
        X_train = IMGs.astype('float32')/255.0
        X_train = 2*X_train-1
        y_train=np.load('tags.npy')

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for i in range(d_iter):
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs, labels = X_train[idx], y_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, 100))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, labels])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 119, batch_size).reshape(-1, 1)
            #print(sampled_labels.shape)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            if epoch%1000==999:
                self.generator.save_weights('model/wgan_g_{}.h5'.format(epoch))
                self.combined.save_weights('model/wgan_c_{}.h5'.format(epoch))


            


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100000, batch_size=64)