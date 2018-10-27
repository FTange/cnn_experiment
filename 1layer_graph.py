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



def experiment_1layer(x, y, name = "", layer1_nodes = [64], dropout1 = [0.2],
               regularization1 = [0.0001], epochs = [10]):

    filename = get_new_filename('results/' + name, '.csv')

    with open(filename, "a") as f:
        f.write("layer1,dropout1,reg1,epochs,accuracy,f1,time\n")

    epoch = epochs[0]

    for l1 in layer1_nodes:
        for drop1 in dropout1:
            for reg1 in regularization1:
                if reg1 == 'batch':
                    model_layers = [keras.layers.Flatten(input_shape=(32, 32)),
                                    keras.layers.Dense(l1, activation=tf.nn.relu), 
                                    tf.keras.layers.BatchNormalization(),
                                    keras.layers.Dropout(drop1),
                                    keras.layers.Dense(46, activation=tf.nn.softmax)]
                else:
                    model_layers = [keras.layers.Flatten(input_shape=(32, 32)),
                                    keras.layers.Dense(l1, activation=tf.nn.relu, 
                                        kernel_regularizer=keras.regularizers.l2(reg1)),
                                    keras.layers.Dropout(drop1),
                                    keras.layers.Dense(46, activation=tf.nn.softmax)]


                accuracy, f1 = repeated_tests(model_layers, x, y, epochs = epoch)

                t = time.localtime()
                t = '{}:{}'.format(t.tm_hour, t.tm_min)

                with open(filename, "a") as f:
                    f.write("{},{},{},{},{},{},{}\n".format(l1,drop1,reg1,epoch,accuracy, f1,t))

                print("l1: {}, drop1: {}, reg1: {}, epoch: {}, acc: {:.2f}, f1: {:.2f}, time: {}".format(
                    l1,drop1,reg1,epoch,accuracy, f1,t))




x, y = get_train_data()
experiment_1layer(x, y,
           name = "dropout graph"
           layer1_nodes = [64, 128,256, 512, 1024, 2048], 
           dropout1 = [0, 0.2,0.4],
           regularization1 = ['batch'],
           epochs = [50])

experiment_1layer(x, y,
           name = "reg graph"
           layer1_nodes = [64, 128,256, 512, 1024, 2048], 
           dropout1 = [0.2],
           regularization1 = [0.0, 0.0001, 'batch'],
           epochs = [50])
