import numpy as np


class Perceptron:
    def __init__(self, input_size, hidden_layers, output_size, scale=1e-3):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.001
        self.learning_rate_decay = 0.95
        self.epochs = 200
        self.reg = 7e-6

        self.params = {'W1': scale * np.random.randn(input_size, hidden_layers), 'b1': np.zeros(hidden_layers),
                       'W2': scale * np.random.randn(hidden_layers, output_size), 'b2': np.zeros(output_size)}

    def loss(self, X, y=None):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        num_samples = X.shape[0]

        # Forward-propagation
        full_c = X.dot(W1) + b1
        X2 = np.maximum(0, full_c)
        scores = X2.dot(W2) + b2

        if y is None:
            return scores

        scores = scores - np.max(scores, axis=1, keepdims=True)
        scores_exp = np.exp(scores)
        softmax = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        loss = np.sum(-np.log(softmax[np.arange(num_samples), y]))
        loss = loss / num_samples + self.reg * (np.sum(W2 * W2) + np.sum(W1 * W1))

        # Back-propagation
        softmax[np.arange(num_samples), y] -= 1
        softmax = softmax / num_samples

        dW2 = X2.T.dot(softmax)
        d_bias_2 = softmax.sum(axis=0)

        dW1 = softmax.dot(W2.T)
        d_full_c = dW1 * (full_c > 0)
        dW1 = X.T.dot(d_full_c)
        d_bias_1 = d_full_c.sum(axis=0)

        dW1 = dW1 + self.reg * 2 * W1
        dW2 = dW2 + self.reg * 2 * W2

        gradients = {'W1': dW1, 'b1': d_bias_1, 'W2': dW2, 'b2': d_bias_2}

        return loss, gradients

    def train(self, X, y, batch_size=200, verbose=False):
        num_train = X.shape[0]
        loss_history = []

        for i in range(self.epochs):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grads = self.loss(X_batch, y=y_batch)
            loss_history.append(loss)

            self.params['W1'] = self.params['W1'] - self.alpha * grads['W1']
            self.params['b1'] = self.params['b1'] - self.alpha * grads['b1']
            self.params['W2'] = self.params['W2'] - self.alpha * grads['W2']
            self.params['b2'] = self.params['b2'] - self.alpha * grads['b2']

            if verbose:
                print('iteration %d / %d: loss %f train_acc %f' % (i, self.epochs, loss, (self.predict(X_batch) == y_batch).mean()))

            if (loss_history[i] / loss_history[i-1]) - 1 > 0.25 and i > 0:
                self.alpha = self.alpha * self.learning_rate_decay

    def predict(self, X):
        return np.argmax(self.loss(X), axis=1)
