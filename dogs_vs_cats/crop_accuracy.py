import json

import numpy as np
import progressbar
from keras.models import load_model

from dogs_vs_cats.config import dogs_vs_cat_config as config
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from preprocessing.croppreprocessor import CropPreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from utils.ranked import rank5_accuracy

with open(config.DATASET_MEAN, 'r') as f:
    means = json.loads(f.read())

simple_preprocessor = SimplePreprocessor(227, 227)
mean_preprocessor = MeanPreprocessor(means['R'], means['G'], means['B'])
crop_preprocessor = CropPreprocessor(227, 227)
image_to_array_preprocessor = ImageToArrayPreprocessor()

print('[INFO] Loading model...')
model = load_model(config.MODEL_PATH)

print('[INFO] Predicting on test data (no crops)...')
test_generator = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[simple_preprocessor, mean_preprocessor,
                                                                           image_to_array_preprocessor], classes=2)
predictions = model.predict_generator(test_generator.generator(), steps=test_generator.num_images // 64,
                                      max_queue_size=10)

rank1, _ = rank5_accuracy(predictions, test_generator.db['labels'])
print(f'[INFO] rank-1: {rank1 * 100: .2f}%')
test_generator.close()

# Let's now test with oversampling
test_generator = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[mean_preprocessor], classes=2)
predictions = []

widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
progress_bar = progressbar.ProgressBar(maxval=test_generator.num_images // 64, widgets=widgets).start()

for i, (images, labels) in enumerate(test_generator.generator(passes=1)):
    for image in images:
        crops = crop_preprocessor.preprocess(image)
        crops = np.array([image_to_array_preprocessor.preprocess(c) for c in crops], dtype='float32')

        # We need to average the predictions on the crops to get a final prediction.
        prediction = model.predict(crops)
        predictions.append(prediction.mean(axis=0))

    progress_bar.update(i)

progress_bar.finish()
print('[INFO] Predicting on test data (with crops)...')
rank1, _ = rank5_accuracy(predictions, test_generator.db['labels'])
print(f'[INFO] Rank-1: {rank1 * 100:.2f}%')
test_generator.close()
