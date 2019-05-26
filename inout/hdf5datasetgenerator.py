import h5py
import numpy as np
from keras.utils import np_utils


class HDF5DatasetGenerator(object):
    def __init__(self, db_path, batch_size, preprocessors=None, augmenter=None, binarize=True, classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.augmenter = augmenter
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(db_path)
        self.num_images = self.db['labels'].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                print(self.batch_size)
                images = self.db['images'[i: i + self.batch_size]]
                labels = self.db['labels'][i: i + self.batch_size]

                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    processed_images = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        processed_images.append(image)

                    images = np.array(processed_images)

                if self.augmenter is not None:
                    # Assuming it is a Keras ImageDataGenerator.
                    images, labels = next(self.augmenter.flow(images, labels, batch_size=self.batch_size))

                yield images, labels

            epochs += 1

    def close(self):
        self.db.close()
