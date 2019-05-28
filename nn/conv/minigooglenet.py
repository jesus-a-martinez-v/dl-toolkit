from keras import backend as K, Input, Model
from keras.layers import concatenate, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization


class MiniGoogLeNet(object):
    @staticmethod
    def conv_module(x, K, k_X, k_Y, stride, channel_dimension, padding='same'):
        # Define a CONV => RELU => BN patten
        x = Conv2D(K, (k_X, k_Y), strides=stride, padding=padding)(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=channel_dimension)(x)

        return x

    @staticmethod
    def inception_module(x, num_K_1x1, num_K_3x3, channel_dimension):
        # Define two CONV modules, then concatenate across the channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, num_K_1x1, 1, 1, (1, 1), channel_dimension)
        conv_3x3 = MiniGoogLeNet.conv_module(x, num_K_3x3, 3, 3, (1, 1), channel_dimension)
        x = concatenate([conv_1x1, conv_3x3], axis=channel_dimension)

        return x

    @staticmethod
    def downsample_module(x, K, channel_dimension):
        # Define the CONV module and POOL, then concatenate across the channel dimensions
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), channel_dimension, padding='valid')
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=channel_dimension)

        return x

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        channel_dimension = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dimension = 1

        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), channel_dimension)

        x = MiniGoogLeNet.inception_module(x, 32, 32, channel_dimension)
        x = MiniGoogLeNet.inception_module(x, 32, 48, channel_dimension)
        x = MiniGoogLeNet.downsample_module(x, 80, channel_dimension)

        x = MiniGoogLeNet.inception_module(x, 112, 48, channel_dimension)
        x = MiniGoogLeNet.inception_module(x, 96, 64, channel_dimension)
        x = MiniGoogLeNet.inception_module(x, 80, 80, channel_dimension)
        x = MiniGoogLeNet.inception_module(x, 48, 96, channel_dimension)
        x = MiniGoogLeNet.downsample_module(x, 96, channel_dimension)

        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dimension)
        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dimension)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs, x, name='googlenet')

        return model
