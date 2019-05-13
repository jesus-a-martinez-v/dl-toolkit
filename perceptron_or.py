import numpy as np

from nn.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

print('[INFO] Training perceptron...')
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=100)

print('[INFO] Testing perceptron...')

for x, target in zip(X, y):
    prediction = p.predict(x)
    print(f'[INFO] data={x}, ground-truth={target[0]}, prediction={prediction}')
