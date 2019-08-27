import json

import cv2
import numpy as np
import progressbar
from sklearn.model_selection import train_test_split

from mx_imagenet_alexnet.config import imagenet_alexnet_config as config
from utils.imagenethelper import ImageNetHelper

print('[INFO] Loading image paths...')
imagenet_helper = ImageNetHelper(config)
train_paths, train_labels = imagenet_helper.build_training_set()
val_paths, val_labels = imagenet_helper.build_validation_set()

print('[INFO] Constructing splits...')
split = train_test_split(train_paths,
                         train_labels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=train_labels,
                         random_state=42)
train_paths, test_paths, train_labels, test_labels = split

datasets = [
    ('train', train_paths, train_labels, config.TRAIN_MX_LIST),
    ('val', val_paths, val_labels, config.VAL_MX_LIST),
    ('test', test_paths, test_labels, config.TEST_MX_LIST)
]

R, G, B = [], [], []

for dataset_type, paths, labels, output_path in datasets:
    print(f'[INFO] Building {output_path}...')
    with open(output_path, 'w') as f:
        widgets = ['Building List: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        progress_bar = progressbar.ProgressBar(maxval=len(paths)).start()

        for index, (path, label) in enumerate(zip(paths, labels)):
            row = '\t'.join([str(index), str(label), path])
            f.write(f'{row}\n')

            if dataset_type == 'train':
                image = cv2.imread(path)
                b, g, r = cv2.mean(image)[:3]

                R.append(r)
                G.append(g)
                B.append(b)

            progress_bar.update(index)

        progress_bar.finish()

print('[INFO} Serializing means...')
D = {'R': np.mean(R),
     'G': np.mean(G),
     'B': np.mean(B)}

with open(config.DATASET_MEAN, 'w') as f:
    f.write(json.dump(D, f))
