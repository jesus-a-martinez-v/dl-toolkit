import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to the output loss/accuracy plot.')

arguments = vars(argument_parser.parse_args())

print('[INFO] Accessing MNIST...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

print('[INFO] Training network...')
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

print('[INFO] Evaluating network...')
predictions = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in label_binarizer.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(arguments['output'])
