from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Sequential


class SRCNN(object):
    @staticmethod
    def build(width, height, depth):
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # The entire SRCNN architecture consists of three CONV => RELU layers with NO zero-padding.
        model.add(Conv2D(64, (9, 9), kernel_initializer='he_normal', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (1, 1), kernel_initializer='he_normal'))
        model.add(Conv2D(depth, (5, 5), kernel_initializer='he_normal'))
        model.add(Activation('relu'))

        return model