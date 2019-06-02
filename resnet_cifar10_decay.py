import matplotlib

from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.resnet import ResNet

matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os

sys.setrecursionlimit(5000)

NUM_EPOCHS = 100
INIT_LR = 1e-1


def polynomial_decay(epoch):
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0

    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    return alpha


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-m', '--model', required=True, help='Path to the output model.')
argument_parser.add_argument('-o', '--output', required=True, help='Path to output directory (logs, plots, etc.)')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading CIFAR-10 data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float')
X_test = X_test.astype('float')

mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

augmenter = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')

figure_path = os.path.sep.join([arguments['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([arguments['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(figure_path, json_path=json_path),
             LearningRateScheduler(polynomial_decay)]

print('[INFO] Compiling model...')
optimizer = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), regularization=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training network...')
model.fit_generator(augmenter.flow(X_train, y_train, batch_size=128),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // 128, epochs=NUM_EPOCHS,
                    callbacks=callbacks, verbose=1)

print('[INFO] Serializing network...')
model.save(arguments['model'])
