from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


class DCGAN(object):
    @staticmethod
    def build_generator(dimension, depth, channels=1, input_dimension=100, output_dimension=512):
        model = Sequential()
        input_shape = (dimension, dimension, depth)
        channel_dimension = -1

        # First set of FC => RELU => BN layers
        model.add(Dense(input_dim=input_dimension, units=output_dimension))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # Second set of FC => RELU => BN layers, this time preparing the number of FC nodes to be reshaped into a volume
        model.add(Dense(dimension * dimension * depth))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # Reshape the output of the previous layer set, upsample + apply a transposed convolution, RELU and BN.
        model.add(Reshape(input_shape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dimension))

        # Apply another upsample and transposed convolution, but this time output the TANH activation
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same'))
        model.add(Activation('tanh'))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        model = Sequential()
        input_shape = (height, width, depth)

        # First set of CONV => RELU layers
        model.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2), input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        # Second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))

        # Fist (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        # Sigmoid layer outputting a single value
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model
