import argparse
import pickle

import h5py

from utils.ranked import rank5_accuracy

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--db', required=True, help='Path to the HDF5 database.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to pre-trained model.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading pre-trained model...')

with open(arguments['model'], 'rb') as f:
    model = pickle.loads(f.read())

db = h5py.File(arguments['db'], 'r')
i = int(db['labels'].shape[0] * 0.75)

print('[INFO] Predicting...')
predictions = model.predict_proba(db['features'][i:])
rank1, rank5 = rank5_accuracy(predictions, db['labels'][i:])

print(f'[INFO] rank-1: {rank1 * 100:.2f}%')
print(f'[INFO] rank-5: {rank5 * 100:.2f}%')

db.close()
