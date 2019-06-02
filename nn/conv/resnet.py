from keras import backend as K
from keras.layers import Flatten, ZeroPadding2D, MaxPooling2D
from keras.layers import Input
from keras.layers import add
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


class ResNet(object):
    @staticmethod
    def residual_module(data, K, stride, channel_dimension, reduce=False, regularization=0.0001,
                        batch_normalization_epsilon=2e-5, batch_normalization_momentum=0.9):
        # The shortcut branch of the ResNet module should be initialized as the input (identity) data
        shortcut = data

        # The first block of the Resnet module are the 1x1 CONVs
        batch_normalization_1 = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                                                   momentum=batch_normalization_momentum)(data)
        activation_1 = Activation('relu')(batch_normalization_1)
        convolution_1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(regularization))(
            activation_1)

        # The second block of the ResNet modile are the 3x3 CONVs.
        batch_normalization_2 = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                                                   momentum=batch_normalization_momentum)(convolution_1)
        activation_2 = Activation('relu')(batch_normalization_2)
        convolution_2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False,
                               kernel_regularizer=l2(regularization))(activation_2)

        # The third block of the ResNet module is another set of 1x1 CONVs.
        batch_normalization_3 = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                                                   momentum=batch_normalization_momentum)(convolution_2)
        activation_3 = Activation('relu')(batch_normalization_3)
        convolution_3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(regularization))(activation_3)

        # If we are to reduce the spatial size, apply a CONV layer to the shortcut
        if reduce:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(regularization))(
                activation_1)

        x = add([convolution_3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, regularization=0.0001, batch_normalization_epsilon=2e-5,
              batch_normalization_momentum=0.9, dataset='cifar'):
        input_shape = (height, width, depth)
        channel_dimension = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dimension = 1

        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                               momentum=batch_normalization_momentum)(inputs)

        if dataset == 'cifar':
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(regularization))(x)
        elif dataset == 'tiny_imagenet':
            # Appy CONV => BN => ACT => POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias=False, padding='same', kernel_regularizer=l2(regularization))(x)
            x = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                                   momentum=batch_normalization_momentum)(x)
            x = Activation('relu')(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(len(stages)):
            # Initialize the stride, then apply a residual module used to reduce the spatial size of the input volume.
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, channel_dimension, reduce=True,
                                       batch_normalization_epsilon=batch_normalization_epsilon,
                                       batch_normalization_momentum=batch_normalization_momentum)

            # Loop over the number of layers in the stage.
            for j in range(stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), channel_dimension,
                                           batch_normalization_epsilon=batch_normalization_epsilon,
                                           batch_normalization_momentum=batch_normalization_momentum)

        x = BatchNormalization(axis=channel_dimension, epsilon=batch_normalization_epsilon,
                               momentum=batch_normalization_momentum)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(regularization))(x)
        x = Activation('softmax')(x)

        model = Model(inputs, x, name='resnet')

        return model
