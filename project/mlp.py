import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
l, w, h = x_train.shape
x_train = x_train.reshape(l, w*h)
x, y, z = x_test.shape
x_test = x_test.reshape(x, y*z)

x_train, x_test = x_train/255.0, x_test/255.0

X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=1,
                                                      stratify=y_train)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_valid_std = sc.transform(X_valid)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1, max_iter=120)
clf.fit(X_train_std, y_train)

y_pred = clf.predict(X_valid_std)
print('Accuracy: %.3f' % accuracy_score(y_valid, y_pred))