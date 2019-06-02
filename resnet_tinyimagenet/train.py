import matplotlib

from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.conv.resnet import ResNet
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor

matplotlib.use('Agg')

from resnet_tinyimagenet.config import tiny_imagenet_config as config
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD
import keras.backend as K
import argparse
import json
import sys

sys.setrecursionlimit(5000)

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--checkpoints', required=True, help='Path to output checkpoint directory.')
argument_parser.add_argument('-m', '--model', type=str, help='Path to *specific* model checkpoint to load.')
argument_parser.add_argument('-s', '--start-epoch', type=int, default=0, help='Epoch to restart training at.')
arguments = vars(argument_parser.parse_args())

augmenter = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.15, horizontal_flip=True, fill_mode='nearest')

with open(config.DATASET_MEAN, 'r') as f:
    means = json.load(f)

simple_preprocessor = SimplePreprocessor(64, 64)
mean_preprocessor = MeanPreprocessor(means['R'], means['G'], means['B'])
image_to_array_preprocessor = ImageToArrayPreprocessor()

train_generator = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, augmenter=augmenter,
                                       preprocessors=[simple_preprocessor, mean_preprocessor,
                                                      image_to_array_preprocessor], classes=config.NUM_CLASSES)
validation_generator = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[simple_preprocessor, mean_preprocessor,
                                                                                image_to_array_preprocessor],
                                            classes=config.NUM_CLASSES)

if arguments['model'] is None:
    print('[INFO] Compiling model...')
    model = ResNet.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6), (64, 128, 256, 512), regularization=0.0005,
                         dataset='tiny_imagenet')
    optimizer = SGD(lr=1e-1, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
else:
    print(f'[INFO] Loading {arguments["model"]}')
    model = load_model(arguments['model'])

    print(f'[INFO] Old learning rate: {K.get_value(model.optimizer.lr)}')
    K.set_value(model.optimizer.lr, 1e-5)
    print(f'[INFO] New learning rate: {K.get_value(model.optimizer.lr)}')

callbacks = [
    EpochCheckpoint(arguments['checkpoints'], every=5, start_at=arguments['start_epoch']),
    TrainingMonitor(config.FIGURE_PATH, json_path=config.JSON_PATH, start_at=arguments['start_epoch'])
]

model.fit_generator(train_generator.generator(), steps_per_epoch=train_generator.num_images // 64,
                    validation_data=validation_generator.generator(),
                    validation_steps=validation_generator.num_images // 64, epochs=50, max_queue_size=10,
                    callbacks=callbacks, verbose=1)

train_generator.close()
validation_generator.close()
