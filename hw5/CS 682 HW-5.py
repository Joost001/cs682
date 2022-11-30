import random
import numpy as np
from data_process import get_CIFAR10_data
import math
from scipy.spatial import distance
from models import KNN, Perceptron, SVM, Softmax
from kaggle_submission import output_submission_csv

# You can change these numbers for experimentation
# For submission we will use the default values
TRAIN_IMAGES = 49000
VAL_IMAGES = 1000
TEST_IMAGES = 5000

data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


def get_acc(pred, y_test):
    return np.sum(y_test == pred)/len(y_test)*100


### Train SVM
print("------SVM------")
svm = SVM()
svm.train(X_train, y_train, verbose=True)

pred_svm = svm.predict(X_train)
print('The training accuracy is given by : %f' % (get_acc(pred_svm, y_train)))

### Validate SVM
pred_svm = svm.predict(X_val)
print('The validation accuracy is given by : %f' % (get_acc(pred_svm, y_val)))

### Test SVM
pred_svm = svm.predict(X_test)
print('The testing accuracy is given by : %f' % (get_acc(pred_svm, y_test)))

### SVM Kaggle Submission
output_submission_csv('svm_submission.csv', svm.predict(X_test))

# ### Train Perceptron
# print("------Perceptron------")
# percept_ = Perceptron(input_size=3072, hidden_layers=50, output_size=10, scale=1e-4)
# percept_.train(X_train, y_train, verbose=False)
#
# pred_percept = percept_.predict(X_train)
# print('The training accuracy is given by : %f' % (get_acc(pred_percept, y_train)))
#
# ### Validation
# pred_percept = percept_.predict(X_val)
# print('The validation accuracy is given by : %f' % (get_acc(pred_percept, y_val)))
#
# ### Test Perceptron
# pred_percept = percept_.predict(X_test)
# print('The testing accuracy is given by : %f' % (get_acc(pred_percept, y_test)))
#
# ### Perceptron Kaggle Submission
# output_submission_csv('perceptron_submission.csv', percept_.predict(X_test))
#
# ### Train Softmax
# print("------SoftMax------")
# softmax = Softmax()
# softmax.train(X_train, y_train, verbose=False)
#
# pred_softmax = softmax.predict(X_train)
# print('The training accuracy is given by : %f' % (get_acc(pred_softmax, y_train)))
#
# ### Validate Softmax
# pred_softmax = softmax.predict(X_val)
# print('The validation accuracy is given by : %f' % (get_acc(pred_softmax, y_val)))
#
# ### Testing Softmax
# pred_softmax = softmax.predict(X_test)
# print('The testing accuracy is given by : %f' % (get_acc(pred_softmax, y_test)))
#
# ### Softmax Kaggle Submission
# output_submission_csv('softmax_submission.csv', softmax.predict(X_test))