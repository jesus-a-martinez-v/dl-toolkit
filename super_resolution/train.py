import matplotlib

from inout.hdf5datasetgenerator import HDF5DatasetGenerator
from nn.conv.srcnn import SRCNN

matplotlib.use('Agg')

from super_resolution.config import sr_config as config
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def super_res_generator(input_data_generator, target_data_generator):
    while True:
        input_data = next(input_data_generator)[0]
        target_data = next(target_data_generator)[0]

        yield (input_data, target_data)


inputs = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUTS_DB, config.BATCH_SIZE)

print('[INFO] Compiling model...')
optimizer = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIMENSION, height=config.INPUT_DIMENSION, depth=3)
model.compile(loss='mse', optimizer=optimizer)

H = model.fit_generator(super_res_generator(inputs.generator(), targets.generator()),
                        steps_per_epoch=inputs.num_images // config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=1)

print('[INFO] Serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history['loss'], label='loss')
plt.title('Loss on super resolution training')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.savefig(config.PLOT_PATH)

inputs.close()
targets.close()
