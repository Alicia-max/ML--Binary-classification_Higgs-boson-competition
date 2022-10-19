import csv
import numpy as np
import random

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


