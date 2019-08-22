import argparse

import cv2
import numpy as np
from keras.models import load_model
from scipy import misc

from super_resolution.config import sr_config as config

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to input image.')
argument_parser.add_argument('-b', '--baseline', required=True, help='Path to baseline image.')
argument_parser.add_argument('-o', '--output', required=True, help='Path to output image.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading model...')
model = load_model(config.MODEL_PATH)

print('[INFO] Generating image...')
image = cv2.imread(arguments['image'])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

scaled = misc.imresize(image, config.SCALE / 1.0, interp='bicubic')
cv2.imwrite(arguments['baseline'], scaled)

output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]

for y in range(0, h - config.INPUT_DIMENSION + 1, config.LABEL_SIZE):
    for x in range(0, w - config.INPUT_DIMENSION + 1, config.LABEL_SIZE):
        crop = scaled[y: y + config.INPUT_DIMENSION, x: x + config.INPUT_DIMENSION]

        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
        output[y + config.PAD: y + config.PAD + config.LABEL_SIZE,
        x + config.PAD: x + config.PAD + config.LABEL_SIZE] = P

output = output[config.PAD: h - ((h % config.INPUT_DIMENSION) + config.PAD),
         config.PAD: w - ((w % config.INPUT_DIMENSION) + config.PAD)]
output = np.clip(output, 0, 255).astype('uint8')

cv2.imwrite(arguments['output'], output)
