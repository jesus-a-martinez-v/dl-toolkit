from keras import backend as K
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


class DeeperGoogLeNet(object):
    @staticmethod
    def conv_module(x, K, k_X, k_Y, stride, channel_dimension, padding='same', regularization=0.0005, name=None):
        convolution_name = None
        batch_normalization_name = None
        activation_name = None

        if name is not None:
            convolution_name = f'{name}_conv'
            batch_normalization_name = f'{name}_bn'
            activation_name = f'{name}_act'

        # Define a CONV => ELU => BN pattern
        x = Conv2D(K, (k_X, k_Y), strides=stride, padding=padding, kernel_regularizer=l2(regularization),
                   name=convolution_name)(x)
        x = Activation('elu', name=activation_name)(x)
        x = BatchNormalization(axis=channel_dimension, name=batch_normalization_name)(x)

        return x

    @staticmethod
    def inception_module(x, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_1x1_proj, channel_dimension,
                         stage, regularization=0.0005):
        # Define the first branch of the Inception module, which consists of 1x1 convolutions.
        first = DeeperGoogLeNet.conv_module(x, num_1x1, 1, 1, (1, 1), channel_dimension, regularization=regularization,
                                            name=f'{stage}_first')

        # Define the second branch of the Inception module, which consists of 1x1 and 3x3 convolutions.
        second = DeeperGoogLeNet.conv_module(x, num_3x3_reduce, 1, 1, (1, 1), channel_dimension,
                                             regularization=regularization, name=f'{stage}_second1')
        second = DeeperGoogLeNet.conv_module(second, num_3x3, 3, 3, (1, 1), channel_dimension,
                                             regularization=regularization, name=f'{stage}_second2')

        # Define the third branch of the Inception module, which are our 1x1 and 5x5 convolutions.
        third = DeeperGoogLeNet.conv_module(x, num_5x5_reduce, 1, 1, (1, 1), channel_dimension,
                                            regularization=regularization, name=f'{stage}_third1')
        third = DeeperGoogLeNet.conv_module(third, num_5x5, 5, 5, (1, 1), channel_dimension,
                                            regularization=regularization, name=f'{stage}_third2')

        # Define the fourth branch of the Inception module, which is the POOL projection.
        fourth = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=f'{stage}_pool')(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num_1x1_proj, 1, 1, (1, 1), channel_dimension,
                                             regularization=regularization, name=f'{stage}_fourth')

        x = concatenate([first, second, third, fourth], axis=channel_dimension, name=f'{stage}_mixed')

        return x

    @staticmethod
    def build(width, height, depth, classes, regularization=0.0005):
        input_shape = (height, width, depth)
        channel_dimension = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dimension = 1

        # Define the model input, followed by a sequence of CONV => POOL => (CONV * 2) => POOL layers
        inputs = Input(shape=input_shape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), channel_dimension, regularization=regularization,
                                        name='block1')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), channel_dimension, regularization=regularization,
                                        name='block2')
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), channel_dimension, regularization=regularization,
                                        name='block3')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool2')(x)

        # Apply two Inception modules followed by a POOL
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, channel_dimension, '3a',
                                             regularization=regularization)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, channel_dimension, '3b',
                                             regularization=regularization)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool3')(x)

        # Apply five Inception modules followed by POOL
        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, channel_dimension, '4a',
                                             regularization=regularization)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, channel_dimension, '4b',
                                             regularization=regularization)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, channel_dimension, '4c',
                                             regularization=regularization)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, channel_dimension, '4d',
                                             regularization=regularization)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, channel_dimension, '4e',
                                             regularization=regularization)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool4')(x)

        # Apply a POOL layer (average) followed by dropout.
        x = AveragePooling2D((4, 4), name='pool5')(x)
        x = Dropout(0.4, name='do')(x)

        # Softmax classifier
        x = Flatten(name='flatten')(x)
        x = Dense(classes, kernel_regularizer=l2(regularization), name='labels')(x)
        x = Activation('softmax', name='softmax')(x)

        # Create the model
        model = Model(inputs, x, name='googlenet')

        return model
