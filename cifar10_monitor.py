import argparse
import os

import matplotlib
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer

from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.minivggnet import MiniVGGNet

matplotlib.use('Agg')

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to the output directory.')
arguments = vars(argument_parser.parse_args())

print(f'[INFO] Process ID: {os.getpid()}')

print('[INFO] Loading CIFAR-10 data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('[INFO] Compiling model...')
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

figure_path = os.path.sep.join([arguments['output'], f'{os.getpid()}.png'])
json_path = os.path.sep.join([arguments['output'], f'{os.getpid()}.json'])
callbacks = [TrainingMonitor(figure_path, json_path)]

print('[INFO] Training network...')
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100, callbacks=callbacks)
