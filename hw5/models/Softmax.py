import numpy as np
from math import inf


class Softmax:
    def __init__(self):
        self.W = None
        self.alpha = 0.5
        self.epochs = 100
        self.reg_const = 0.05
        self.learning_rate_decay = 0.9
    
    def calc_gradient(self, X, y):
        num_features = X.shape[1]
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))

        if self.W is None:
            self.W = np.random.normal(loc=0.0, scale=1e-3, size=(num_features, num_classes))

        b = np.zeros(num_classes)

        scores = np.dot(X, self.W) + b
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))

        scores_ = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        scores_[np.arange(num_samples), y] -= 1
        scores_ = scores_ / num_samples
        dW = X.T.dot(scores_)

        loss = 0.5 * self.reg_const * np.sum(self.W * self.W)
        dW += self.reg_const * self.W

        return loss, dW
    
    def train(self, X_train, y_train, verbose=False):
        last_loss = inf
        step_size = self.alpha
        for e in range(self.epochs):
            loss, dW = self.calc_gradient(X_train, y_train)
            if verbose:
                print("SOFTMAX: The current epoch: " + str(e) + " is " + str(loss))
            self.W = self.W - step_size * dW
            if loss < last_loss:
                last_loss = loss
            else:
                last_loss = loss
                step_size = step_size * self.learning_rate_decay
        return self.W

    def predict(self, X_test):
        return X_test.dot(self.W).argmax(axis=1)
