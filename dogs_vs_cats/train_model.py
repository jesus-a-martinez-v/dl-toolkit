import argparse
import pickle

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

argument_parse = argparse.ArgumentParser()
argument_parse.add_argument('-d', '--db', required=True, help='Path to HDF5 database.')
argument_parse.add_argument('-m', '--model', required=True, help='Path to output model.')
argument_parse.add_argument('-j', '--jobs', type=int, default=-1,
                            help='Number of jobs to run when tuning hyperparameters.')
arguments = vars(argument_parse.parse_args())

db = h5py.File(arguments['db'], 'r')
split_index = int(db['labels'].shape[0] * 0.75)

print('[INFO] Tuning hyperparameters...')
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(solver='lbfgs', multi_class='auto'), params, cv=3, n_jobs=arguments['jobs'])
model.fit(db['features'][:split_index], db['labels'][:split_index])
print(f'[INFO] Best hyperparameters: {model.best_params_}')

print('[INFO] Evaluating...')
predictions = model.predict(db['features'][split_index:])
print(classification_report(db['labels'][split_index:], predictions, target_names=db['label_names']))

accuracy = accuracy_score(db['labels'][split_index:], predictions)
print(f'[INFO] Score: {accuracy}')

print('[INFO] Saving model...')
with open(arguments['model'], 'wb') as f:
    f.write(pickle.dumps(model.best_estimator_))

db.close()
