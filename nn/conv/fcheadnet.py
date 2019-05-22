from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten


class FCHeadNet(object):
    @staticmethod
    def build(base_model, classes, dense_neurons):
        head_model = base_model.output
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(dense_neurons, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)

        head_model = Dense(classes, activation='softmax')(head_model)

        return head_model
