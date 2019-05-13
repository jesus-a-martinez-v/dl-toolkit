from keras import backend as K
from keras.layers import Activation, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.models import Sequential


class ShallowNet(object):
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
