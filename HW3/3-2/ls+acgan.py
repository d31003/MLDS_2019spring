from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import sys
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

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
        #losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
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
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss='mse',
            optimizer=optimizer)


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

    def build_discriminator(self):

        discriminator=Sequential()
        
        discriminator.add(Conv2D(32, kernel_size=4, input_shape=self.img_shape))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(64, kernel_size=4, padding='same'))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(128, kernel_size=4))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(256, kernel_size=4))
        discriminator.add(LeakyReLU())
        discriminator.add(Flatten())
        #discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = discriminator(img)

        # Determine validity and label of the image
        validity = Dense(1)(features)#, activation="sigmoid"
        label = Dense(1)(features)#self.num_classes, activation="softmax"

        return Model(img, [validity, label])

    '''
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        #model_input = multiply([img, label_embedding])

        validity = discriminator(model_input)

        return Model([img, label], validity)
    '''
    def train(self, epochs, batch_size=128):

        

        IMGs=np.load('real_images.npy')
        X_train = IMGs.astype('float32')/255.0
        X_train = 2*X_train-1
        y_train=np.load('tags.npy')

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            sampled_labels = np.random.randint(0, 119, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Generate a half batch of new images
            #gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            #sampled_labels = np.random.randint(0, 119, batch_size).reshape(-1, 1)
            #print(sampled_labels.shape)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            if epoch%1000==999:
                self.generator.save_weights('model/ls+acgan_g_{}.h5'.format(epoch))
                self.combined.save_weights('model/ls+acgan_c_{}.h5'.format(epoch))


            


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=50000, batch_size=64)