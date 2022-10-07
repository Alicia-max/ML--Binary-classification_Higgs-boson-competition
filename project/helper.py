# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    features= np.genfromtxt(data_path, delimiter=",", skip_header=1)[:,2:]
    ID = np.genfromtxt(data_path, delimiter=",", skip_header=1)[:, 0].astype(np.int)


    # sub-sample
    if sub_sample:
        y = yb[::50]
        features= features[::50]
        ID = ids[::50]

    return y, features, ID
