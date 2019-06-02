import argparse

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to the output model.')

arguments = vars(argument_parser.parse_args())

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))

simple_preprocessor = SimplePreprocessor(32, 32)
image_to_array_preprocessor = ImageToArrayPreprocessor()

simple_dataset_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor, image_to_array_preprocessor])
data, labels = simple_dataset_loader.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.25, random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print('[INFO] Compiling model...')
optimizer = SGD(lr=.005)

model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=100, verbose=1)

print('[INFO] Serializing network...')
model.save(arguments['model'])

print('[INFO] Evaluating network...')
predictions = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
