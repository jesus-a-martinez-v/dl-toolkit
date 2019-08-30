import argparse
import json
import logging
import os

import mxnet as mx

from mx_imagenet_squeezenet.config import imagenet_squeezenet_config as config
from nn.mxconv.mxsqueezenet import MxSqueezeNet

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--checkpoints', required=True, help='Path to output checkpoint directory.')
argument_parser.add_argument('-p', '--prefix', required=True, help='Name of model prefix.')
argument_parser.add_argument('-s', '--start-epoch', type=int, default=0, help='Epoch to restart training at.')
arguments = vars(argument_parser.parse_args())

logging.basicConfig(level=logging.DEBUG, filename=f'training_{arguments["start_epoch"]}.log', filemode='w')

with open(config.DATASET_MEAN, 'r') as f:
    means = json.loads(f.read())

batch_size = config.BATCH_SIZE * config.NUM_DEVICES

train_iter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC,
                                   data_shape=(3, 227, 227),
                                   batch_size=batch_size,
                                   rand_crop=True,
                                   rand_mirror=True,
                                   rotate=15,
                                   max_shear_ratio=0.1,
                                   mean_r=means['R'],
                                   mean_g=means['G'],
                                   mean_b=means['B'],
                                   preprocess_threads=config.NUM_DEVICES * 2)

val_iter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC,
                                 data_shape=(3, 227, 227),
                                 batch_size=batch_size,
                                 mean_r=means['R'],
                                 mean_g=means['G'],
                                 mean_b=means['B'])

opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=0.0005,
                       rescale_grad=1.0 / batch_size)  # TODO For some reason, MXNET does not work with this.

checkpoints_path = os.path.sep.join([arguments['checkpoints'], arguments['prefix']])
argument_params = None
auxiliary_params = None

if arguments['start_epoch'] <= 0:
    print('[INFO] Building network...')
    model = MxSqueezeNet.build(config.NUM_CLASSES)
else:
    print(f'[INFO] Loading epoch {arguments["start_epoch"]}...')
    model = mx.model.FeedForward.load(checkpoints_path, arguments['start_epoch'])

    argument_params = model.arg_params
    auxiliary_params = model.aux_params
    model = model.symbol

model = mx.model.FeedForward(ctx=[mx.gpu(i) for i in range(config.NUM_DEVICES)],
                             symbol=model,
                             initializer=mx.initializer.Xavier(),
                             arg_params=argument_params,
                             aux_params=auxiliary_params,
                             # optimizer=opt,
                             num_epoch=90,
                             begin_epoch=arguments['start_epoch'],
                             optimizer='sgd',
                             # Below are the parameters for the optimizer.
                             learning_rate=1e-2,
                             momentum=0.9,
                             wd=0.0002)
                             # rescale_grad=1.0 / batch_size)

batch_end_callbacks = [mx.callback.Speedometer(batch_size, 250)]
epoch_end_callbacks = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

print('[INFO] Training network...')
model.fit(X=train_iter,
          eval_data=val_iter,
          eval_metric=metrics,
          batch_end_callback=batch_end_callbacks,
          epoch_end_callback=epoch_end_callbacks)
