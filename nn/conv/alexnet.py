from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2


class AlexNet(object):
    @staticmethod
    def build(width, height, depth, classes, regularization=0.0002):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dimension = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dimension = 1

        # Block #1: First CONV => RELU => POOL layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='same',
                         kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: Second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: First set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #4: Second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(regularization)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(regularization)))
        model.add(Activation('softmax'))

        return model
