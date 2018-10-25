import numpy as np
import pickle
import os.path
import tensorflow as tf
from tensorflow import keras
import time
import itertools as it

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

def experiment_1layer_cnn(x, y, layer1_nodes = [256],
                          dropout1 = [0.2], epochs = [10], filters = [32], 
                          kernel_size = [(3, 3)], strides = [(1, 1)],
                          pool_size = [(2, 2)]):

    filename = get_new_filename('results/nn_1_cnn_1_hidden', '.csv')

    with open(filename, "a") as f:
        f.write("layer1,dropout1,epochs,filters,kernel_size,strides,pool_size,accuracy,f1,time\n")

    parameters = it.product(layer1_nodes, dropout1, epochs, filters, kernel_size, strides, pool_size)

    for l1, drop1, epoch, num_filters, k_size, stride, p_size in parameters:

        model_layers = [tf.keras.layers.Conv2D(num_filters, kernel_size = k_size, strides = stride, 
                                               activation = 'relu', input_shape = (32, 32, 1)),
                        tf.keras.layers.MaxPooling2D(pool_size = p_size, padding='valid',
                                                     strides=None), # can be done, kernel or something else
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dropout(drop1),
                        tf.keras.layers.Dense(l1, activation=tf.nn.relu),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dense(46, activation=tf.nn.softmax)]


        accuracy, f1 = repeated_tests(model_layers, x, y, epochs = epoch)

        t = time.localtime()
        t = '{}:{}'.format(t.tm_hour, t.tm_min)

        with open(filename, "a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(l1,l2,drop1,epoch,num_filters,k_size,stride,p_size,accuracy,f1,t))

        print("l1: {}, drop1: {}, epoch: {}, filters: {}, kernel_size: {}, stride: {}, pool_size: {} acc: {:.2f}, f1: {:.2f}, time: {}".format(
            l1,drop1,epoch, num_filters, k_size, stride, p_size, accuracy, f1,t))


x, y = get_train_data()
x = x.reshape(-1, 32, 32, 1)

experiment_1layer_cnn(x, y,
           layer1_nodes = [256, 512, 1024],
           layer2_nodes = [256],
           dropout1 = [0.2],
           epochs = [50],
           filters = [32, 64], 
           kernel_size = [(3, 3)], 
           strides = [(1, 1), (2, 2), None],
           pool_size = [(2, 2), (3, 3)])
