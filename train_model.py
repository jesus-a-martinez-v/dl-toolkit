import argparse
import pickle

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--db', required=True, help='Path to HDF5 database.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to output model.')
argument_parser.add_argument('-j', '--jobs', type=int, default=-1,
                             help='Number of jobs to run when tuning hyperparameters.')
arguments = vars(argument_parser.parse_args())

db = h5py.File(arguments['db'], 'r')
i = int(db['labels'].shape[0] * 0.75)  # Split index.

print('[INFO] Tuning hyperparameters...')
parameters = {'C': (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)}
model = GridSearchCV(LogisticRegression(solver='lbfgs', multi_class='auto'), parameters, cv=5, n_jobs=arguments['jobs'])
model.fit(db['features'][:i], db['labels'][:i])
print(f'[INFO] Best hyperparameters: {model.best_params_}')

print('[INFO] Evaluating...')
predictions = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], predictions, target_names=db['label_names']))

print('[INFO] Saving model...')
with open(arguments['model'], 'wb') as f:
    f.write(pickle.dumps(model.best_estimator_))

db.close()
