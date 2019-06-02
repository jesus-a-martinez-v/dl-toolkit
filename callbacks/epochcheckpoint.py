import os

from keras.callbacks import Callback


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        super(Callback, self).__init__()

        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs=None):
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path, f'Epoch_{self.int_epoch + 1}.hdf5'])
            self.model.save(p, overwrite=True)

        self.int_epoch += 1
