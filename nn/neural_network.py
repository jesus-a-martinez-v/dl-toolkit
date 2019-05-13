import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers, alpha=0.1):
        self.W = list()
        self.layers = layers
        self.alpha = alpha

        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return f'NeuralNetwork: {"-".join(str(l) for l in self.layers)}'

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch={epoch + 1}, loss={loss}')

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        for layer in range(len(self.W)):
            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmoid_derivative(A[-1])]

        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        D = D[::-1]

        for layer in range(len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in range(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
