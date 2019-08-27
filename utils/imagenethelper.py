import os

import numpy as np


class ImageNetHelper(object):
    def __init__(self, config):
        self.config = config

        self.label_mappings = self.build_class_labels()

    def build_class_labels(self):
        with open(self.config.WORD_IDS, 'r') as f:
            rows = f.read().strip().split('\n')
            label_mappings = {}

            for row in rows:
                word_id, label, human_readable_label = row.split(' ')
                label_mappings[word_id] = int(label) - 1

        return label_mappings

    def build_training_set(self):
        with open(self.config.TRAIN_LIST, 'r') as f:
            rows = f.read().strip().split('\n')

            paths = []
            labels = []

            for row in rows:
                partial_path, image_num = row.strip().split(' ')

                path = os.path.sep.join([self.config.IMAGES_PATH, 'train', f'{partial_path}.JPEG'])
                word_id = partial_path.split('/')[0]
                label = self.label_mappings[word_id]

                paths.append(path)
                labels.append(label)

        return np.array(paths), np.array(labels)

    def build_validation_set(self):
        paths = []
        labels = []

        with open(self.config.VAL_LIST, 'r') as f:
            val_file_names = f.read()
            val_file_names = val_file_names.strip().split('\n')

        with open(self.config.VAL_LABELS, 'r') as f:
            val_labels = f.read()
            val_labels = val_labels.strip().split('\n')

        for row, label in zip(val_file_names, val_labels):
            partial_path, image_num = row.strip().split(' ')

            path = os.path.sep.join([self.config.IMAGES_PATH, 'val', f'{partial_path}.JPEG'])
            paths.append(path)
            labels.append(int(label) - 1)

        return np.array(paths), np.array(labels)
