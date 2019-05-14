import argparse
import os
import random

import numpy as np
import progressbar
from imutils import paths
from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

from inout.hdf5datasetwriter import HDF5DatasetWriter

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to the input dataset.')
argument_parser.add_argument('-o', '--output', required=True, help='Path to output HDF5 file.')
argument_parser.add_argument('-b', '--batch-size', type=int, default=32,
                             help='Batch size of images to be passed through the network.')
argument_parser.add_argument('-s', '--buffer-size', type=int, default=1000,
                             help='Size of the feature extraction buffer.')
arguments = vars(argument_parser.parse_args())

batch_size = arguments['batch_size']

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))
random.shuffle(image_paths)

labels = [p.split(os.path.sep)[-2] for p in image_paths]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print('[INFO] Loading network...')
model = VGG16(weights='imagenet', include_top=False)

dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7), arguments['output'], data_key='features',
                            buffer_size=arguments['buffer_size'])
dataset.store_class_labels(label_encoder.classes_)

widgets = ['Extracting Features: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
progress_bar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i: i + batch_size]
    batch_labels = labels[i: i + batch_size]
    batch_images = list()

    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batch_images.append(image)

    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    dataset.add(features, batch_labels)
    progress_bar.update(i)

dataset.close()
progress_bar.finish()
