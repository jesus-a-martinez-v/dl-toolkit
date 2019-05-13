from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.neural_network import NeuralNetwork

print('[INFO] Loading MNIST (sample) dataset...')
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())

print(f'[INFO] Samples: {data.shape[0]}, dim: {data.shape[1]}')

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=.25)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print('[INFO] Training network...')
nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
print(f'[INFO] {nn}')
nn.fit(X_train, y_train, epochs=1000)

print('[INFO] Evaluating network...')
predictions = nn.predict(X_test)
predictions = predictions.argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), predictions))
