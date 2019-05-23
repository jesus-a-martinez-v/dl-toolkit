import argparse
import glob
import os

import numpy as np
from keras.datasets import cifar10
from keras.engine.saving import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-m', '--models', required=True, help='Path to models directory.')
arguments = vars(argument_parser.parse_args())

X_test, y_test = cifar10.load_data()[1]  # Only the test set.
X_test = X_test.astype('float') / 255.0

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_binarizer = LabelBinarizer()
y_test = label_binarizer.fit_transform(y_test)

model_paths = os.path.sep.join([arguments['models'], '*.model'])
model_paths = list(glob.glob(model_paths))
models = list()

for i, model_path in enumerate(model_paths):
    print(f'[INFO] Loading model {i + 1}/{len(model_paths)}')
    models.append(load_model(model_path))

print('[INFO] Evaluating ensemble...')
predictions = list()

for model in models:
    predictions.append(model.predict(X_test, batch_size=64))

predictions = np.average(predictions, axis=0)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))
