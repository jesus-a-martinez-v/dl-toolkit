import matplotlib

from callbacks.trainingmonitor import TrainingMonitor
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.conv.alexnet import AlexNet
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

matplotlib.use('Agg')

from dogs_vs_cats.config import dogs_vs_cat_config as config
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

augmenter = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.15, horizontal_flip=True, fill_mode='nearest')

with open(config.DATASET_MEAN, 'r') as f:
    means = json.loads(f.read())

simple_preprocessor = SimplePreprocessor(227, 227)
patch_preprocessor = PatchPreprocessor(227, 227)
mean_preprocessor = MeanPreprocessor(means['R'], means['G'], means['B'])
image_to_array_preprocessor = ImageToArrayPreprocessor()

train_generator = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, augmenter=augmenter,
                                       preprocessors=[patch_preprocessor, mean_preprocessor,
                                                      image_to_array_preprocessor], classes=2)
validation_generator = HDF5DatasetGenerator(config.VAL_HDF5, 128, preprocessors=[simple_preprocessor, mean_preprocessor,
                                                                                 image_to_array_preprocessor],
                                            classes=2)

print('[INFO] Compiling model...')
optimizer = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, regularization=0.0002)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

path = os.path.sep.join([config.OUTPUT_PATH, f'{os.getpid()}.png'])
callbacks = [TrainingMonitor(path)]

model.fit_generator(train_generator.generator(),
                    steps_per_epoch=train_generator.num_images // 128,
                    validation_data=validation_generator.generator(),
                    validation_steps=validation_generator.num_images // 128,
                    epochs=75,
                    max_queue_size=5,
                    callbacks=callbacks,
                    verbose=1)

print('[INFO] Serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

train_generator.close()
validation_generator.close()
