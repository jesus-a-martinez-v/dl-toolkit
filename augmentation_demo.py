import argparse

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to the input image.')
argument_parser.add_argument('-o', '--output', required=True,
                             help='Path to output directory to store augmentation examples.')
argument_parser.add_argument('-p', '--prefix', type=str, default='image', help='Output filename prefix.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading example image...')
image = load_img(arguments['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

augmenter = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                               zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

total = 0

print('[INFO] Generating images...')
image_generator = augmenter.flow(image, batch_size=1, save_to_dir=arguments['output'], save_prefix=arguments['prefix'],
                                 save_format='jpg')

for image in image_generator:
    total += 1

    if total == 10:
        break
