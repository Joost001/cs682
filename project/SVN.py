import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
l, w, h = x_train.shape
x_train = x_train.reshape(l, w*h)
x, y, z = x_test.shape
x_test = x_test.reshape(x, y*z)

def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

x_train, x_test = x_train/255.0, x_test/255.0

X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1,
                                                      stratify=y_train)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

svm = SVC(kernel='linear', random_state=1, C=0.1)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_valid_std)
print('Accuracy: %.3f' % accuracy_score(y_valid, y_pred))