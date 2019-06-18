import os

INPUT_IMAGES = '../assets/datasets/ukbench/ukbench100'

BASE_OUTPUT = 'output'
IMAGES = os.path.sep.join([BASE_OUTPUT, 'images'])
LABELS = os.path.sep.join([BASE_OUTPUT, 'labels'])

INPUTS_DB = os.path.sep.join([BASE_OUTPUT, 'inputs.hdf5'])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, 'outputs.hdf5'])

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'srcnn.model'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])

BATCH_SIZE = 128
NUM_EPOCHS = 10

SCALE = 2.0
INPUT_DIMENSION = 33

LABEL_SIZE = 21
PAD = int((INPUT_DIMENSION - LABEL_SIZE) / 2.0)

STRIDE = 14