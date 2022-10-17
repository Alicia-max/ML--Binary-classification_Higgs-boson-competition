import csv
import numpy as np
import random


def set_seed(seed):
    """
    Function to fix the random number generator.
    For reproducability purposes.
    """
    np.random.seed(seed)
    random.seed(seed)

def sigmoid(x):
    """
    TODO
    """
    return np.exp(x)/(1+np.exp(x))


def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error. 
    inputs are the targeted y, the sample matrix tx and the feature vector w. 
    """
    e = y - tx.dot(w)
    mse = 1/2*np.mean(e**2)
    return mse

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient of the MSE loss
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_gradient_logistic(y, tx, w):
    """
    TODO and check implementation
    """
    grad = tx.T.dot(sigmoid(tx.dot(w))-y) 
    err = 1/2*np.mean(sigmoid(tx.dot(w))-y)
    return grad, err


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    features= np.genfromtxt(data_path, delimiter=",", skip_header=1)[:,2:]
    ID = np.genfromtxt(data_path, delimiter=",", skip_header=1)[:, 0].astype(np.int)

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    yb[np.where(y=='s')] = 1
    y = yb

    # sub-sample for testing purposes
    if sub_sample:
        y = y[::50]
        features= features[::50]
        ID = ID[::50]

    return y, features, ID

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]