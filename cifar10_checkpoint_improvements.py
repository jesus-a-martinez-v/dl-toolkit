import argparse
import os

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer

from nn.conv.minivggnet import MiniVGGNet

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-w', '--weights', required=True, help='Path to weights directory.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading CIFAR-10 data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

print('[INFO] Compiling model...')
optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

file_name = os.path.sep.join([arguments['weights'], 'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', mode='min', save_best_only=True)
callbacks = [checkpoint]

print('[INFO] Training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=40, callbacks=callbacks)
