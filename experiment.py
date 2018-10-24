import numpy as np
import pickle
import os.path
import tensorflow as tf
from tensorflow import keras
import time

from sklearn.model_selection import train_test_split

from evaluate import *
from read_data import *

def get_new_filename(filename, ending):
    if os.path.exists(filename + ending):
        i = 2
        while os.path.exists(filename + str(i) + '.csv'):
            i += 1
        return filename + str(i) + ending
    return filename + ending

def experiment_1layer(x, y, layer1_nodes = [64], dropout1 = [0.2],
               regularization1 = [0.0001], epochs = [10]):

    filename = get_new_filename('results/nn_2_layer', '.csv')

    with open(filename, "a") as f:
        f.write("layer1,dropout1,reg1,epochs,accuracy,f1,time\n")

    for l1 in layer1_nodes:
        for drop1 in dropout1:
            for reg1 in regularization1:
                for epoch in epochs:
                    model_layers = [keras.layers.Flatten(input_shape=(32, 32)),
                                    keras.layers.Dense(l1, activation=tf.nn.relu, 
                                        kernel_regularizer=keras.regularizers.l2(reg1)),
                                    keras.layers.Dropout(drop1),
                                    keras.layers.Dense(46, activation=tf.nn.softmax)]


                    accuracy, f1 = repeated_tests(model_layers, x, y, epochs = epochs)

                    t = time.localtime()
                    t = '{}:{}'.format(t.tm_hour, t.tm_min)

                    with open(filename, "a") as f:
                        f.write("{},{},{},{},{},{},{}\n".format(l1,drop1,reg1,epoch,accuracy, f1,t))

                    print("l1: {}, drop1: {}, reg1: {}, epoch: {}, acc: {:.2f}, f1: {:.2f}, time: {}".format(
                        l1,drop1,reg1,epoch,accuracy, f1,t))


def experiment_2layer(x, y, layer1_nodes = [64], layer2_nodes = [64], 
               dropout1 = [0.2], dropout2 = [0.2], 
               regularization1 = [0.0001], regularization2 = [0.0001], epochs = [10]):

    filename = get_new_filename('results/nn_2_layer', '.csv')

    with open(filename, "a") as f:
        f.write("layer1,layer2,dropout1,dropout2,reg1,reg2,epochs,accuracy,f1,time\n")

    for l1 in layer1_nodes:
        for l2 in layer2_nodes:
            for drop1 in dropout1:
                for drop2 in dropout2:
                    for reg1 in regularization1:
                        for reg2 in regularization2:
                            for epoch in epochs:
                                model_layers = [keras.layers.Flatten(input_shape=(32, 32)),
                                                keras.layers.Dense(l1, activation=tf.nn.relu, 
                                                    kernel_regularizer=keras.regularizers.l2(reg1)),
                                                keras.layers.Dropout(drop1),
                                                keras.layers.Dense(l2, activation=tf.nn.relu, 
                                                    kernel_regularizer=keras.regularizers.l2(reg2)),
                                                keras.layers.Dropout(drop2),
                                                keras.layers.Dense(46, activation=tf.nn.softmax)]


                                accuracy, f1 = repeated_tests(model_layers, x, y, epochs = epochs)

                                t = time.localtime()
                                t = '{}:{}'.format(t.tm_hour, t.tm_min)

                                with open(filename, "a") as f:
                                    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(l1,l2,drop1,drop2,reg1,reg2,epoch,accuracy, f1,t))

                                print("l1: {}, l2: {}, drop1: {}, drop2: {}, reg1: {}, reg2: {}, epoch: {}, acc: {:.2f}, f1: {:.2f}, time: {}".format(
                                    l1,l2,drop1,drop2,reg1,reg2,epoch,accuracy, f1,t))


x, y = get_train_data()
experiment_1layer(x, y,
           layer1_nodes = [64, 128,256, 512], 
           dropout1 = [0.2,0.4],
           regularization1 = [0.0, 0.0001, 0.001],
           epochs = [10, 20, 40, 60])
        
# experiment_2layer(x, y)


"""
with open('Devanagari_data.pkl', 'rb') as f:
    data, labels = pickle.load(f)

input_dim = data.shape[1] * data.shape[2]
output_dim = labels.max() + 1

data = data / 255.0

x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.2, random_state=1)

# experiment(x_train, y_train)

experiment(x_train, y_train, "results/2_layers",
           layer1_nodes = [128,256, 512], 
           layer2_nodes = [128,256, 512], 
           dropout1 = [0.2,0.4],
           dropout2 = [0.2,0.4], 
           regularization1 = [0.0, 0.0001, 0.001],
           regularization2 = [0.0, 0.0001, 0.001])
"""
