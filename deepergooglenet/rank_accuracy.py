import json

from keras.models import load_model

from deepergooglenet.config import tiny_imagenet_config as config
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from utils.ranked import rank5_accuracy

with open(config.DATASET_MEAN, 'r') as f:
    means = json.load(f)

simple_preprocessor = SimplePreprocessor(64, 64)
mean_preprocessor = MeanPreprocessor(means['R'], means['G'], means['B'])
image_to_array_preprocessor = ImageToArrayPreprocessor()

test_generator = HDF5DatasetGenerator(config.TEST_HDF5, 64,
                                      preprocessors=[simple_preprocessor,
                                                     mean_preprocessor,
                                                     image_to_array_preprocessor],
                                      classes=config.NUM_CLASSES)

print('[INFO] Loading model...')
model = load_model(config.MODEL_PATH)

print('[INFO] Predicting on test data...')
predictions = model.predict_generator(test_generator.generator(), steps=test_generator.num_images // 64,
                                      max_queue_size=10)

rank1, rank5 = rank5_accuracy(predictions, test_generator.db['labels'])
print(f'[INFO] Rank-1:  {rank1 * 100:.2f}%')
print(f'[INFO] Rank-5:  {rank5 * 100:.2f}%')

test_generator.close()
