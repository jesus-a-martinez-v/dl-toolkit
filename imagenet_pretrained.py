import argparse

import cv2
import numpy as np
from keras.applications import VGG16, VGG19, InceptionV3, Xception, ResNet50, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to the input image.')
argument_parser.add_argument('-m', '--model', type=str, default='vgg16', help='Name of the pre-trained network to use.')
arguments = vars(argument_parser.parse_args())

MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'inception': InceptionV3,
    'xception': Xception,
    'resnet': ResNet50
}

if arguments['model'] not in MODELS.keys():
    raise AssertionError(f'The --model command line argument should be one of {", ".join(MODELS.keys())}')

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if arguments['model'] in ('inception', 'xception'):
    input_shape = (299, 299)
    preprocess = preprocess_input

print(f'[INFO] Loading {arguments["model"]}...')
Network = MODELS[arguments['model']]
model = Network(weights='imagenet')

print('[INFO] Loading and pre-processing image...')
image = load_img(arguments['image'], target_size=input_shape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess(image)

print(f'[INFO] Classifying image with "{arguments["model"]}"')
predictions = model.predict(image)
P = imagenet_utils.decode_predictions(predictions)

for i, (imagenet_id, label, prob) in enumerate(P[0]):
    print(f'{i + 1}. {label}: {prob * 100:.2f}%')

original = cv2.imread(arguments['image'])
imagenet_id, label, prob = P[0][0]

cv2.putText(original, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Classification', original)
cv2.waitKey(0)
