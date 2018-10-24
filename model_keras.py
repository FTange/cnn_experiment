import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn.model_selection import train_test_split

import plot

""" cross validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 5, shuffle = True)
for train_idxs, test_idxs in kfold.split(data, labels):
    train, test = (data[train_idxs, ], labels[test_idxs])
"""



with open('Devanagari_data.pkl', 'rb') as f:
    data, labels = pickle.load(f)

input_dim = data.shape[1] * data.shape[2]
output_dim = labels.max() + 1

"""
labels_matrix = np.zeros((labels.shape[0], output_dim))
for idx, label in enumerate(labels):
    labels_matrix[idx, label-1] = 1
"""

# data = data.reshape((-1, input_dim))

data = data / 255.0

x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.2, random_state=1)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    # keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation=tf.nn.relu),
    # keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(output_dim, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_split = 0.2, verbose = 1)

# print(history.history['val_acc'])

_, test_acc = model.evaluate(x_test, y_test)
print(test_acc)

# plot.plot_history([("standard", history)], key = "acc")
