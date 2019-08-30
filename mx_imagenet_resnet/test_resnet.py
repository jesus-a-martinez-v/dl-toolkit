import argparse
import json
import os

import mxnet as mx

from mx_imagenet_resnet.config import imagenet_resnet_config as config

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--checkpoints', required=True, help='Path to output checkpoints directory.')
argument_parser.add_argument('-p', '--prefix', required=True, help='Name of model prefix.')
argument_parser.add_argument('-e', '--epoch', type=int, required=True, help='Epoch number to load.')
arguments = argument_parser.parse_args()

with open(config.DATASET_MEAN, 'r') as f:
    means = json.loads(f.read())

test_iter = mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC,
                                  data_shape=(3, 224, 224),
                                  batch_size=config.BATCH_SIZE,
                                  mean_r=means['R'],
                                  mean_g=means['G'],
                                  mean_b=means['B'])

print('[INFO] Loading model...')
checkpoints_path = os.path.sep.join([arguments['checkpoints'], arguments['prefix']])
model = mx.model.FeedForward.load(checkpoints_path, arguments['epoch'])

model = mx.model.FeedForward(ctx=[mx.gpu(0)], symbol=model.symbol, arg_params=model.arg_params,
                             aux_params=model.aux_params)

print('[INFO] Predicting on test data...')
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
rank_1, rank_5 = model.score(test_iter, eval_metric=metrics)

print(f'[INFO] Rank-1: {rank_1 * 100:.2f}%')
print(f'[INFO] Rank-5: {rank_5 * 100:.2f}%')
