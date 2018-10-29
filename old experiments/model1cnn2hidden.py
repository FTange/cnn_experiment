import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

import plot
from read_data import *
from evaluate import *


x, y = get_train_data()
x_test, y_test = get_test_data()
# x = x / x.max()

x = x.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)

input_dim = x.shape[1] * x.shape[2]
output_dim = y.max() + 1

model_layers = [tf.keras.layers.Conv2D(32, kernel_size = (3, 3), strides = (1, 1), 
                                       activation = 'relu', input_shape = (32, 32, 1)),
                tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid',
                                             strides=(1, 1)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Dense(46, activation=tf.nn.softmax)]

model = tf.keras.models.Sequential(model_layers)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=100)
y_pred = model.predict(x_test)
y_pred = np.apply_along_axis(np.argmax, 1, np.round(y_pred))

p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average = 'weighted')
acc = accuracy_score(y_test, y_pred)

print('precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, accuracy: {:.4f}'.format(p, r, f, acc))

with open('predictions_1cnn_2hidden', 'wb') as f:
    pickle.dump(y_pred, f)


# accuracy, f1 = repeated_tests(model_layers, x_train, y_train, y_train_nn)
