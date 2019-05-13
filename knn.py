import argparse

from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to the input dataset.')
argument_parser.add_argument('-k', '--neighbors', type=int, default=1,
                             help='Number of nearest neighbors for classification.')
argument_parser.add_argument('-j', '--jobs', type=int, default=-1,
                             help='Number of jobs for k-NN distance (-1 uses all available cores).')

arguments = vars(argument_parser.parse_args())

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))

simple_preprocessor = SimplePreprocessor(32, 32)
simple_data_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor])
data, labels = simple_data_loader.load(image_paths, verbose=5000)
data = data.reshape((data.shape[0], 3072))

print(f'[INFO] features matrix: {data.nbytes / (1024 * 1024.0)}MB')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=.25, random_state=42)

print('[INFO] evaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors=arguments['neighbors'], n_jobs=arguments['jobs'])
model.fit(train_X, train_y)

print(classification_report(test_y, model.predict(test_X), target_names=label_encoder.classes_))
