import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def crossval(model_layers, x, y, num_folds = 5, epochs = 10,
             optimizer = tf.train.AdamOptimizer(), 
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy']):

    kfold = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1)
    cvscores = []

    model = keras.Sequential(model_layers)
    model.compile(optimizer = optimizer, 
                  loss = loss,
                  metrics = metrics)
    # save model weights to reinitialize the model in each k-fold
    Wsave = model.get_weights()

    for i, (k_train, k_test) in enumerate(kfold.split(x, y)):

        model.set_weights(Wsave) # reset the weights / untrain the model
        model.fit(x[k_train], y[k_train], epochs = epochs, verbose = 0) 

        _, test_acc = model.evaluate(x[k_test], y[k_test], verbose = 0)
        cvscores.append(test_acc)

    # print(cvscores)

    return sum(cvscores) / len(cvscores)



def repeated_tests(model_layers, x, y, num_tests = 2, epochs = 10):

    acc_scores = np.empty(num_tests)
    f1_scores = np.empty(num_tests)

    model = keras.Sequential(model_layers)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # hack such that weights will be initialized
    model.fit(x[0:1], y[0:1], epochs = 1, verbose = 0)
    # save model weights to reinitialize the model in each k-fold
    Wsave = model.get_weights()
    
    kfold = StratifiedShuffleSplit(n_splits=num_tests, test_size=.20, random_state=0)

    for i, (k_train, k_test) in enumerate(kfold.split(x, y)):

        model.set_weights(Wsave) # reset the weights / untrain the model
        model.fit(x[k_train], y[k_train], epochs = epochs, verbose = 0) 

        pred = model.predict(x[k_test], verbose = 0)
        pred = np.apply_along_axis(np.argmax, 1, np.round(pred))
        
        acc_scores[i] = accuracy_score(y[k_test], pred)
        f1_scores[i] = f1_score(y[k_test], pred, average = "weighted")

    return acc_scores.mean(), f1_scores.mean()



if __name__ == "__main__":
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


    model_layers = [keras.layers.Flatten(input_shape=(32, 32)),
                    keras.layers.Dense(64, activation=tf.nn.relu),
                    # keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
                    # keras.layers.Dropout(0.2),
                    keras.layers.Dense(64, activation=tf.nn.relu),
                    # keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
                    # keras.layers.Dropout(0.2),
                    keras.layers.Dense(output_dim, activation=tf.nn.softmax)]

    crossval(model_layers, x_train, y_train)
