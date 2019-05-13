import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def predict(X, W):
    predictions = sigmoid_activation(X.dot(W))

    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0] = 1

    return predictions


def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i: i + batch_size], y[i: i + batch_size]


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-e', '--epochs', type=float, default=100, help='Number of epochs.')
argument_parser.add_argument('-a', '--alpha', type=float, default=.01, help='Learning rate.')
argument_parser.add_argument('-b', '--batch-size', type=int, default=64, help='Size of SGD mini batch.')

arguments = vars(argument_parser.parse_args())

X, y = make_blobs(n_samples=10000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones((X.shape[0]))]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.5, random_state=42)

print('[INFO] Training...')
W = np.random.randn(X.shape[1], 1)
losses = list()

for epoch in range(arguments['epochs']):
    epoch_loss = list()

    for batch_X, batch_y in next_batch(train_X, train_y, arguments['batch_size']):
        predictions = sigmoid_activation(batch_X.dot(W))

        error = predictions - batch_y
        loss = np.sum(error ** 2)
        epoch_loss.append(loss)

        derivative = error * sigmoid_derivative(predictions)
        gradient = batch_X.T.dot(derivative)

        W -= arguments['alpha'] * gradient

    loss = np.average(epoch_loss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f'[INFO] Epoch={int(epoch + 1)}, loss={loss}')

print('[INFO] Evaluating.')
predictions = predict(test_X, W)
print(classification_report(test_y, predictions))

plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(test_X[:, 0], test_X[:, 1], marker='o', c=test_y[:, 0], s=30)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, arguments['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
