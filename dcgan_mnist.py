import argparse
import os

import cv2
import numpy as np
from imutils import build_montages
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle

from nn.conv.dcgan import DCGAN

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to output directory.')
argument_parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train for.')
argument_parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size for training.')
arguments = vars(argument_parser.parse_args())

NUMBER_OF_EPOCHS = arguments['epochs']
BATCH_SIZE = arguments['batch_size']

print('[INFO] Loading MNIST dataset...')
(X_train, _), (X_test, _) = mnist.load_data()
train_images = np.concatenate([X_train, X_test])

# Scale into the range [-1, 1], which is the range of the tanh function.
train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype('float') - 127.5) / 127.5

print('[INFO] Building generator...')
generator = DCGAN.build_generator(7, 64, channels=1)

print('[INFO] Building discriminator...')
discriminator = DCGAN.build_discriminator(28, 28, 1)
discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUMBER_OF_EPOCHS)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

print('[INFO] Building GAN...')
discriminator.trainable = False
gan_input = Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUMBER_OF_EPOCHS)
gan.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

print('[INFO] Starting training...')
benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

for epoch in range(NUMBER_OF_EPOCHS):
    print(f'[INFO] Starting epoch {epoch + 1}/{NUMBER_OF_EPOCHS}')
    batches_per_epoch = int(train_images.shape[0] / BATCH_SIZE)

    for i in range(batches_per_epoch):
        output_path = None

        image_batch = train_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        generated_images = generator.predict(noise, verbose=0)

        X = np.concatenate((image_batch, generated_images))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        (X, y) = shuffle(X, y)

        discriminator_loss = discriminator.train_on_batch(X, y)

        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        gan_loss = gan.train_on_batch(noise, [1] * BATCH_SIZE)

        if i == batches_per_epoch - 1:
            output_path = [arguments['output'], f'epoch_{str(epoch + 1).zfill(4)}_output.png']
        else:
            if epoch < 10 and i % 25 == 0:
                output_path = [arguments['output'], f'epoch_{str(epoch + 1).zfill(4)}_step_{str(i).zfill(5)}.png']
            elif epoch >= 10 and i % 100 == 0:
                output_path = [arguments['output'], f'epoch_{str(epoch + 1).zfill(4)}_step_{str(i).zfill(5)}.png']

        if output_path is not None:
            print(
                f'[INFO] Step {epoch + 1}_{i}: discriminator_loss={discriminator_loss:.6f}, adversarial_loss={gan_loss:.6f}')

            images = generator.predict(benchmark_noise)
            images = ((images * 127.5) + 127.5).astype('uint8')
            images = np.repeat(images, 3, axis=-1)
            visualization = build_montages(images, (28, 28), (16, 16))[0]

            output_path = os.path.sep.join(output_path)
            cv2.imwrite(output_path, visualization)
