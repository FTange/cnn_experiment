import os, imageio, re, pickle
import numpy as np

from sklearn.model_selection import train_test_split

"""
If called multiple times with different folders assumes that every folder contains the same number of subdirectories and in the same order
"""
def read_directory(folder):
    data = []
    target = []
    target_names = []
    for idx, (root, dirs, files) in enumerate(os.walk(folder)):
        if root == folder: # don't go through start directory
            root_number = idx
            continue

        target_names.append(re.sub('.*/', '', root))
        for f in files:
            img = imageio.imread(root + '/' + f)
            data.append(img)
            target.append(idx)

    # root directory is skipped so subtract all names afterwards by 1
    data = np.array(data) # turn data into 3 dimensional array
    target = np.array([i-1 if i > root_number else i for i in target])

    return (data, target)

def get_train_data():
    with open('Devanagari_train.pkl', 'rb') as f:
        x_train, y_train = pickle.load(f)
    return x_train / x_train.max(), y_train
    
def get_test_data():
    with open('Devanagari_test.pkl', 'rb') as f:
        x_test, y_test = pickle.load(f)
    return x_test / x_test.max(), y_test


def transform_y_for_nn(y, classes = None):
    if classes is None:
        classes = y.max()+1
    y_nn = np.zeros((y.shape[0], classes))
    for idx, example in enumerate(y):
        y_nn[idx, example] = 1
    return y_nn


if __name__ == "__main__":
    data_train, target_train = read_directory('../DevanagariHandwrittenCharacterDataset/Train')
    data_test, target_test = read_directory('../DevanagariHandwrittenCharacterDataset/Test')

    data_combined = np.vstack((data_test, data_train))
    target_commbined = np.hstack((target_test, target_train))

    with open('Devanagari_train.pkl', 'wb') as f:
        pickle.dump((data_train, target_train), f)

    with open('Devanagari_test.pkl', 'wb') as f:
        pickle.dump((data_test, target_test), f)

    # with open('Devanagari_data.pkl', 'rb') as f:
        # d, t = pickle.load(f)

# data, target and target_names to files


# things to test

# how does our model perform as we change the architecture / depth of model
# how does our model perform as we change the number of categories
# how does our model perform as we change the size of the training set - how large does it need to be to perform 'well'
# find the best model for each number of hidden layers
