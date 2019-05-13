import argparse

import cv2
import numpy as np
from imutils import paths
from keras.engine.saving import load_model

from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to the pre-trained model.')

arguments = vars(argument_parser.parse_args())

class_labels = ['cat', 'dog', 'panda']

print('[INFO] Sampling images...')
image_paths = np.array(list(paths.list_images(arguments['dataset'])))
indices = np.random.randint(0, len(image_paths), size=(10,))
image_paths = image_paths[indices]

simple_preprocessor = SimplePreprocessor(32, 32)
image_to_array_preprocessor = ImageToArrayPreprocessor()

simple_dataset_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor, image_to_array_preprocessor])
data, labels = simple_dataset_loader.load(image_paths)
data = data.astype('float') / 255.0

print('[INFO] Loading pre-trained network...')
model = load_model(arguments['model'])

print('[INFO] Predicting...')
predictions = model.predict(data, batch_size=64).argmax(axis=1)

for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    cv2.putText(image, f'Label: {class_labels[predictions[i]]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
