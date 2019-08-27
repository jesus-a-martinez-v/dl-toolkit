import json
import os

import cv2
import numpy as np
import progressbar
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dogs_vs_cats.config import dogs_vs_cat_config as config
from inout.hdf5datasetwriter import HDF5DatasetWriter
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor

train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [p.split(os.path.sep)[-1].split('.')[0] for p in train_paths]

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

split = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels,
                         random_state=42)
train_paths, test_paths, train_labels, test_labels = split

split = train_test_split(train_paths, train_labels, test_size=config.NUM_VAL_IMAGES, stratify=train_labels,
                         random_state=42)
train_paths, val_paths, train_labels, val_labels = split

datasets = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

aspect_aware_preprocessor = AspectAwarePreprocessor(256, 256)
R = list()
G = list()
B = list()

for d_type, paths, labels, output_path in datasets:
    print(f'[INFO] Building {output_path}...')
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), output_path)

    widgets = ['Building Dataset: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    progress_bar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aspect_aware_preprocessor.preprocess(image)

        if d_type == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        progress_bar.update(i)

    progress_bar.finish()
    writer.close()

print('[INFO] Serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}

with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dumps(D))
