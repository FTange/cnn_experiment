import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import plot
from read_data import *
from evaluate import *


x, y = get_train_data()

x = x.reshape(-1, 32, 32, 1)

input_dim = x.shape[1] * x.shape[2]
output_dim = y.max() + 1


x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)

y_train_nn = transform_y_for_nn(y_train)

model_layers = [tf.keras.layers.Conv2D(32, kernel_size = (3, 3), strides = (1, 1), 
                                       activation = 'relu', input_shape = (32, 32, 1)),
                tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding='valid',
                                             strides=None) # can be done, pool_size or something else
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)]

model = tf.keras.models.Sequential(model_layers)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
y_pred = model.evaluate(x_test, y_test)

# accuracy, f1 = repeated_tests(model_layers, x_train, y_train, y_train_nn)

print('acc: {}, f1: {}'.format(accuracy, f1))
