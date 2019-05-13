import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.minivggnet import MiniVGGNet
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aspect_aware_preprocessor = AspectAwarePreprocessor(64, 64)
image_to_array_preprocessor = ImageToArrayPreprocessor()

simple_dataset_loader = SimpleDatasetLoader(preprocessors=[aspect_aware_preprocessor, image_to_array_preprocessor])
data, labels = simple_dataset_loader.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print('[INFO] Compiling model...')
optimizer = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('[INFO] Training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100, verbose=1)

print('[INFO] Evaluating network...')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

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
plt.show()
