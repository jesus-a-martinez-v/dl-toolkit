import os
import random
import shutil

import cv2
from imutils import paths
from scipy import misc

from inout.hdf5datasetwriter import HDF5DatasetWriter
from super_resolution.config import sr_config as config

for path in [config.IMAGES, config.LABELS]:
    if not os.path.exists(path):
        os.makedirs(path)

print('[INFO] Creating temporary images...')
image_paths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(image_paths)

total = 0

for image_path in image_paths:
    image = cv2.imread(image_path)

    # Grab the dimensions of the input image and crop the image such that it tiles nicely when we generate the training
    # data and labels.
    (h, w) = image.shape[:2]
    w -= int(w % config.SCALE)
    h -= int(h % config.SCALE)
    image = image[0:h, 0:w]

    # To generate our training images, we first need to downscale the image by the scale factor, and then upscale
    # it back to the original size. This process allows us to generate low resolution inputs we'll then learn to
    # reconstruct the high resolution versions from
    scaled = misc.imresize(image, 1.0 / config.SCALE, interp='bicubic')
    scaled = misc.imresize(scaled, config.SCALE / 1.0, interp='bicubic')

    for y in range(0, h - config.INPUT_DIMENSION + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIMENSION + 1, config.STRIDE):
            crop = scaled[y: y + config.INPUT_DIMENSION, x: x + config.INPUT_DIMENSION]

            target = image[y + config.PAD: y + config.PAD + config.LABEL_SIZE,
                     x + config.PAD: x + config.PAD + config.LABEL_SIZE]

            crop_path = os.path.sep.join([config.IMAGES, f'{total}.png'])
            target_path = os.path.sep.join([config.LABELS, f'{total}.png'])

            cv2.imwrite(crop_path, crop)
            cv2.imwrite(target_path, target)

            total += 1

print('[INFO] Building HDF5 datasets...')
input_paths = sorted(list(paths.list_images(config.IMAGES)))
output_paths = sorted(list(paths.list_images(config.LABELS)))

input_writer = HDF5DatasetWriter((len(input_paths), config.INPUT_DIMENSION, config.INPUT_DIMENSION, 3),
                                 config.INPUTS_DB)
output_writer = HDF5DatasetWriter((len(output_paths), config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUTS_DB)

for input_path, output_path in zip(input_paths, output_paths):
    input_image = cv2.imread(input_path)
    output_image = cv2.imread(output_path)
    input_writer.add([input_image], [-1])
    output_writer.add([output_image], [-1])

input_writer.close()
output_writer.close()

print('[INFO] Cleaning up...')
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)
