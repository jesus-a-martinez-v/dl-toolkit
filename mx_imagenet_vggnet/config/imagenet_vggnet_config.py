from os import path

BASE_PATH = '/home/jesus/Data/ImageNet'

IMAGES_PATH = path.sep.join([BASE_PATH, 'images'])
IMAGE_SETS_PATHS = path.sep.join([BASE_PATH, 'image_sets'])
DEVKIT_PATH = path.sep.join([BASE_PATH, 'devkit', 'ILSVRC2015', 'devkit', 'data'])
WORD_IDS = path.sep.join([DEVKIT_PATH, 'map_clsloc.txt'])

TRAIN_LIST = path.sep.join([IMAGE_SETS_PATHS, 'train.txt'])
VAL_LIST = path.sep.join([IMAGE_SETS_PATHS, 'val.txt'])
VAL_LABELS = path.sep.join([DEVKIT_PATH, 'ILSVRC2015_clsloc_validation_ground_truth.txt'])

NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

MX_OUTPUT = '/home/jesus/Data/ImageNet'
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'train.lst'])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'val.lst'])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, 'lists', 'test.lst'])

TRAIN_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'train.rec'])
VAL_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'val.rec'])
TEST_MX_REC = path.sep.join([MX_OUTPUT, 'rec', 'test.rec'])

# DATASET_MEAN = path.sep.join([BASE_PATH, 'output', 'imagenet_mean.json'])  # TODO Check this!
DATASET_MEAN = './output/imagenet_mean.json'

#### FROM EXPERIMENT TO EXPERIMENT, THE VARIABLES BELOW ARE THE ONLY WE NEED TO CHANGE ####
BATCH_SIZE = 32
NUM_DEVICES = 1
