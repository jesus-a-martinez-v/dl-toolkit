import argparse

from imutils import paths
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset.')

arguments = vars(argument_parser.parse_args())

print('[INFO] Loading images...')
image_paths = list(paths.list_images(arguments['dataset']))

simple_preprocessor = SimplePreprocessor(32, 32)
simple_data_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor])
data, labels = simple_data_loader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=.25, random_state=5)

for r in (None, 'l1', 'l2'):
    print(f'[INFO] Training model with "{r}"" penalty.')
    model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant', tol=1e-3, eta0=0.01,
                          random_state=42)
    model.fit(train_X, train_y)

    accuracy = model.score(test_X, test_y)
    print(f'[INFO] "{r}" penalty accuracy: {accuracy * 100}%')
