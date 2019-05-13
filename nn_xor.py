import numpy as np

from nn.neural_network import NeuralNetwork

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=10000)

for x, target in zip(X, y):
    prediction = nn.predict(x)[0][0]
    step = 1 if prediction > 0.5 else 0
    print(f'[INFO] data={x}, ground-truth={target[0]}, prediction={prediction}, step={step}')
