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
# x = x / x.max()

input_dim = x.shape[1] * x.shape[2]
output_dim = y.max() + 1


x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)

y_train_nn = transform_y_for_nn(y_train)

model_layers = [tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)]

accuracy, f1 = repeated_tests(model_layers, x_train, y_train, y_train_nn)

print('acc: {}, f1: {}'.format(accuracy, f1))



"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)
y_pred = model.evaluate(x_test, y_test)
"""




"""
x = x.reshape((x.shape[0], -1))

inputs = keras.Input(shape=(input_dim,))  # Returns a placeholder tensor

x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(output_dim, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train_nn, batch_size=32, epochs=10)
y_pred = model.predict(x_test)

print(accuracy_score(y_test, np.apply_along_axis(np.argmax, 1, y_pred)))
"""





"""
model = keras.Sequential()
model.add(keras.layers.Dense(input_dim, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(output_dim, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 10, batch_size = 32)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
"""
