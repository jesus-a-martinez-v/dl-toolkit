import argparse

import matplotlib
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from nn.conv.minivggnet import MiniVGGNet

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to the output loss/accuracy plot.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading CIFAR-10 data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('[INFO] Compiling model...')
optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=40)

print('[INFO] Evaluating network...')
predictions = model.predict(X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 40), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 40), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on CIFAR-10')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(arguments['output'])
