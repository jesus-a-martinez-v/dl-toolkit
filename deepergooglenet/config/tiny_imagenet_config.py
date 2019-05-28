from os import path

TRAIN_IMAGES = '../assests/datasets/tiny-imagenet-200/train'
VAL_IMAGES = '../assests/datasets/tiny-imagenet-200/val/images'

VAL_MAPPINGS = '../assests/datasets/tiny-imagenet-200/val/val_annotations.txt'

WORDNET_IDS = '../assests/datasets/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../assests/datasets/tiny-imagenet-200/words.txt'

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

TRAIN_HDF5 = '../assets/datasets/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '../assets/datasets/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '../assets/datasets/tiny-imagenet-200/hdf5/test.hdf5'

DATASET_MEAN = 'output/tiny-imagenet-200-mean.json'

OUTPUT_PATH = 'output'
MODEL_PATH = path.sep.join([OUTPUT_PATH, 'checkpoints/epoch_70.hdf5'])
FIGURE_PATH = path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.png'])
JSON_PATH = path.sep.join([OUTPUT_PATH, 'deepergooglenet_tinyimagenet.json'])
