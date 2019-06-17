import os

import cv2
import numpy as np
from keras import backend as K
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scipy.optimize import fmin_l_bfgs_b


class NeuralStyle(object):
    def __init__(self, settings: dict):
        self.settings = settings

        (w, h) = load_img(self.settings['input_path']).size
        self.dimensions = (h, w)

        self.content = self.preprocess(self.settings['input_path'])
        self.style = self.preprocess(self.settings['style_path'])

        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        # Allocate memory of our output image, then combine the content, style and output into
        # a single tensor to they can be fed through the network.
        self.output = K.placeholder((1, self.dimensions[0], self.dimensions[1], 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        print('[INFO] Loading network...')
        self.model = self.settings['net'](weights='imagenet', include_top=False, input_tensor=self.input)

        # Builds a dictionary that maps the name of each layer inside the network to the actual layer output.
        layer_map = {l.name: l.output for l in self.model.layers}

        content_features = layer_map[self.settings['content_layer']]
        style_features = content_features[0, :, :, :]
        output_features = content_features[2, :, :, :]

        content_loss = self.feature_recon_loss(style_features, output_features)
        content_loss *= self.settings['content_weight']

        style_loss = K.variable(0.0)
        weight = 1.0 / len(self.settings['style_layers'])

        for layer in self.settings['style_layers']:
            style_output = layer_map[layer]

            style_features = style_output[1, :, :, :]
            output_features = style_output[2, :, :, :]

            T = self.style_recon_loss(style_features, output_features)
            style_loss += (weight * T)

        style_loss *= self.settings['style_weight']
        total_variation_loss = self.settings['tv_weight'] * self.total_variation_loss(self.output)
        total_loss = content_loss + style_loss + total_variation_loss

        gradients = K.gradients(total_loss, self.output)
        outputs = [total_loss]
        outputs += gradients

        # The implementation of L-BFGS we use requires that our loss and gradients be two separate functions, so
        # here we are creating a Keras function that can compute both the loss and gradients together and then
        # return each separately using two different class methods
        self.loss_and_grads = K.function([self.output], outputs)

    def preprocess(self, image_path):
        image = load_img(image_path, target_size=self.dimensions)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return image

    def deprocess(self, image):
        image = image.reshape((self.dimensions[0], self.dimensions[1], 3))

        # These are the well-known the mean values across the ImageNet training set. We are undoing
        # the zero-centering here.
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.680

        image = np.clip(image, 0, 255).astype('uint8')

        return image

    @staticmethod
    def gram_matrix(X):
        features = K.permute_dimensions(X, (2, 0, 1))
        features = K.batch_flatten(features)
        features = K.dot(features, K.transpose(features))

        return features

    @staticmethod
    def feature_recon_loss(style_features, output_features):
        return K.sum(K.square(output_features - style_features))

    def style_recon_loss(self, style_features, output_features):
        gram_style = self.gram_matrix(style_features)
        gram_generated = self.gram_matrix(output_features)

        scale = 1.0 / float((2 * 3 * self.dimensions[0] * self.dimensions[1]) ** 2)
        loss = scale * K.sum(K.square(gram_generated - gram_style))

        return loss

    def total_variation_loss(self, X):
        (h, w) = self.dimensions

        # We use array slicing along the borders to avoid border artifacts.
        A = K.square(X[:, : h - 1, : w - 1, :] - X[:, 1:, : w - 1, :])
        B = K.square(X[:, : h - 1, : w - 1, :] - X[:, : h - 1, 1:, :])
        loss = K.sum(K.pow(A + B, 1.25))

        return loss

    def transfer(self, max_evaluations=20):
        X = np.random.uniform(0, 255, (1, self.dimensions[0], self.dimensions[1], 3)) - 128

        for i in range(self.settings['iterations']):
            print(f'[INFO] Starting iteration {i + 1}/{self.settings["iterations"]}...')
            (X, loss, _) = fmin_l_bfgs_b(self.loss, X.flatten(), fprime=self.gradients, maxfun=max_evaluations)
            print(f'[INFO] End of iteration {i + 1}. Loss: {loss:.4e}...')

            image = self.deprocess(X.copy())
            path = os.path.sep.join([self.settings['output_path'], f'iter_{i}.png'])
            cv2.imwrite(path, image)

    def loss(self, X):
        X = X.reshape((1, self.dimensions[0], self.dimensions[1], 3))
        loss_value = self.loss_and_grads([X])[0]

        return loss_value

    def gradients(self, X):
        X = X.reshape((1, self.dimensions[0], self.dimensions[1], 3))
        output = self.loss_and_grads([X])

        return output[1].flatten().astype('float64')
