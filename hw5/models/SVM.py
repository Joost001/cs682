import numpy as np
from math import inf


class SVM:
    def __init__(self):
        self.W = None
        self.alpha = 0.5
        self.epochs = 100
        self.reg_const = .5
        self.learning_rate_decay = 0.8
        
    def calc_gradient(self, X, y):
        if self.W is None:
            self.W = np.random.randint(150, size=(3072, 10)) - 75

        dW = np.zeros(self.W.shape)

        num_samples = X.shape[0]
        num_features = self.W.shape[1]
        loss = 0.0
        for i in range(num_samples):
            scores = X[i].dot(self.W)
            actual_class_score = scores[y[i]]
            for j in range(num_features):
                if j == y[i]:
                    continue
                margin = scores[j] - actual_class_score + 1
                if margin > 0:
                    loss += margin
                    dW[:, y[i]] = dW[:, y[i]] - X[i]
                    dW[:, j] = dW[:, j] + X[i]

        # Calculate average
        loss /= num_samples
        dW = dW / num_samples

        # Regularization
        loss += self.reg_const * np.sum(self.W * self.W)
        dW = dW + self.reg_const * 2 * self.W

        return loss, dW

    def train(self, X_train, y_train, verbose=False):
        last_loss = inf
        step_size = self.alpha
        for e in range(self.epochs):
            loss, dW = self.calc_gradient(X_train, y_train)
            if verbose:
                print("SVM: The current loss at epoch " + str(e) + " is " + str(loss))
            self.W = self.W - step_size * dW
            if loss < last_loss:
                last_loss = loss
            else:
                last_loss = loss
                step_size = step_size * self.learning_rate_decay
        return self.W

    def predict(self, X_test):
        return X_test.dot(self.W).argmax(axis=1)