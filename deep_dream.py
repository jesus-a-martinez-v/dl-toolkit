import argparse

import cv2
import numpy as np
from keras import backend as K
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scipy import ndimage


def preprocess(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def deprocess(image):
    image = image.reshape((image.shape[1], image.shape[2], 3))

    # We need to "undo" the preprocessing done for INception to bring the image back into the [0, 255] range.
    image /= 2.0
    image += 0.5
    image *= 255.0
    image = np.clip(image, 0, 255).astype('uint8')

    # We must conform to OpenCV's channel ordering:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def resize_image(image, size):
    resized = np.copy(image)
    shape = (1, float(size[0]) / resized.shape[1], float(size[1]) / resized.shape[2], 1)
    resized = ndimage.zoom(resized, shape, order=1)

    return resized


def eval_loss_and_gradients(X):
    output = fetch_loss_grads([X])
    (loss, gradients) = (output[0], output[1])

    return loss, gradients


def gradient_ascent(X, iterations, alpha, max_loss=-np.inf):
    for i in range(iterations):
        (loss, gradients) = eval_loss_and_gradients(X)

        if loss > max_loss:
            break

        print(f'[INFO] Loss at {i}: {loss}')
        X += alpha * gradients

    return X

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to input image.')
argument_parser.add_argument('-o', '--output', required=True, help='Path to output (dreamed) image.')
arguments = vars(argument_parser.parse_args())

# Layers and weights we are going to use to make the network "dream". The larger the weight, the larger the contribution of that layer to the
# output dream.
LAYERS = {
    'mixed2': 2.0,
    'mixed3': 0.5
}

# Tweaking these constants will produce different "dreams"
NUMBER_OF_OCTAVES = 3
OCTAVE_SCALE = 1.4
ALPHA = 0.001
NUMBER_OF_ITERATIONS = 50
MAX_LOSS = 10.0

# Tells Keras NOT to update any weights during deep dream
K.set_learning_phase(0)

print('[INFO] Loading inception network...')
model = InceptionV3(weights='imagenet', include_top=False)
dream = model.input

loss = K.variable(0.0)
layer_map = {layer.name: layer for layer in model.layers}

for layer_name in LAYERS:
    x = layer_map[layer_name].output
    coefficient = LAYERS[layer_name]
    scaling = K.prod(K.cast(K.shape(x), 'float32'))

    # We use array slicing here to avoid border artifacts caused by border pixels.
    loss += coefficient * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

gradients = K.gradients(loss, dream)[0]
gradients /= K.maximum(K.mean(K.abs(gradients)), 1e-7)

outputs = [loss, gradients]
fetch_loss_grads = K.function([dream], outputs)

image = preprocess(arguments['image'])
dimensions = image.shape[1:3]

octave_dimensions = [dimensions]

for i in range(1, NUMBER_OF_OCTAVES):
    size = [int(d / (OCTAVE_SCALE ** i)) for  d in dimensions]
    octave_dimensions.append(size)

octave_dimensions = octave_dimensions[::-1]  # Reverse so the smallest dimensions appear first.

original = np.copy(image)
shrunk = resize_image(image, octave_dimensions[0])

for (octave, size) in enumerate(octave_dimensions):
    print(f'[INFO] Starting octave {octave}...')
    image = resize_image(image, size)

    image = gradient_ascent(image, iterations=NUMBER_OF_ITERATIONS, alpha=ALPHA, max_loss=MAX_LOSS)

    upscaled = resize_image(shrunk, size)
    downscaled = resize_image(original, size)

    lost_detail = downscaled - upscaled
    image += lost_detail

    shrunk = resize_image(original, size)  # Make the original the shrunk image so we can repeat the process.

image = deprocess(image)
cv2.imwrite(arguments['output'], image)