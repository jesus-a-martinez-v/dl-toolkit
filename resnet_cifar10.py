import matplotlib
from keras.datasets import cifar10
from keras.engine.saving import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.resnet import ResNet

matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
import numpy as np
import argparse
import sys

sys.setrecursionlimit(5000)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--checkpoints', required=True, help='Path to output checkpoint directory.')
argument_parser.add_argument('-m', '--model', type=str, help='Path to *specific* model checkpoint to load.')
argument_parser.add_argument('-s', '--start-epoch', type=int, default=0, help='Epoch to restart training at.')
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

if arguments['model'] is None:
    print('[INFO] Compiling model...')
    optimizer = SGD(lr=1e-1)

    model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), regularization=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
else:
    print(f'[INFO[ Loading {arguments["model"]}...')
    model = load_model(arguments['model'])

    print(f'[INFO] Old learing rate: {K.get_value(model.optimizer.lr)}')
    K.set_value(model.optimizer.lr, 1e-5)
    print(f'[INFO] New learning rate: {K.get_value(model.optimizer.lr)}')

callbacks = [
    EpochCheckpoint(arguments['checkpoints'], every=5, start_at=arguments['start_epoch']),
    TrainingMonitor('output/resnet56_cifar10.png', json_path='output/resnet56_cifar10.json',
                    start_at=arguments['start_epoch'])
]

print('[INFO] Training network...')
model.fit_generator(augmenter.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // 64, epochs=100, callbacks=callbacks, verbose=1)
