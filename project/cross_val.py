# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from preprocessing import *


def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold  """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k_fold, method, log=False,  **params):
    '''
    Perform k-fold cross-validation for a given method. 
    This method aims to separate the given data set (x and y)
    into k_fold in which there is a validation & a training set used to learn the given method.
    It also supports feature modification (polynomial expansion, standarization, sin/cos transformation, add offset).
    
    Input:
        - y : Expected value vector
        - x : Data Matrix
        - k_indices : The indices of the test set k_indices
        - method : One of the 6 implemented method 
        - log : Boolean that determine the prediction used (predict or predict_log)
        - Method's parameters
    Output : 
        - acc_tr_tmp : Array with train accuracy  for each fold
        - acc_te_tmp : Array with test accuracy for each fold
    ''' 
    
    acc_tr_tmp=[]
    acc_te_tmp=[]
    degree = params['degree']
    fourier=params['fourier']
    
    params_without_degree_offset = params
    del params_without_degree_offset['degree']
    del params_without_degree_offset['fourier']
   
    
    for k in range(k_fold) :
        
        # Get k'th subgroup in test, others in train
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        
        y_te = y[te_indice]
        y_tr = y[tr_indice] 
        x_te = x[te_indice,:]
        x_tr = x[tr_indice,:]
        
        # Form data with polynomial degree
        tx_tr = build_poly(x_tr, degree)
        tx_te = build_poly(x_te, degree)
        
        # Standarization
        std_tx_tr, mean_tx, std_tx = standardize(tx_tr)
        std_tx_te,mean_te, std_te= standardize(tx_te,mean_tx, std_tx)
        
        # Cos & Sin transfo
        std_tx_tr = fourier_encoding(std_tx_tr, fourier)
        std_tx_te = fourier_encoding(std_tx_te, fourier)
        
        # Offset Adding
        std_tx_tr = add_offset(std_tx_tr)
        std_tx_te = add_offset(std_tx_te)
        
        w, loss = method(y_tr, std_tx_tr, **params_without_degree_offset)
        
        # Access accuracy
        if(log) : 
            acc_tr_tmp.append(accuracy(y_tr, predict_log(std_tx_tr,w)))
            acc_te_tmp.append(accuracy(y_te, predict_log(std_tx_te,w)))
            
        else : 
            acc_tr_tmp.append(accuracy(y_tr, predict(std_tx_tr,w)))
            acc_te_tmp.append(accuracy(y_te, predict(std_tx_te,w)))
            
    return acc_tr_tmp, acc_te_tmp

def cross_tunning(y, x, k_fold, method, parameters, seed, log = False) :
    '''
    Tune the received parameters with a k-fold cross validation.
    Input:
        - y : Expected value vector
        - x : Data Matrix
        - k_fold : Number of folds
        - parameters : Dictionnary of parameters to tune
        - method : One of the 6 methods implemented 
        - log : Boolean that determine the prediction used (predict or predict_log)
        - seed : Seed used by the np.random library to define the k_indices
 
    Output : 
        - acc_tr_tmp : Array with the mean train accuracy of each fold for each set of parameters 
        - acc_te_tmp : Array with the mean test accuracy of each fold for each set of parameters 
        - std_tr, std_te :  Array with the standard deviation of the train accuracy of each fold for each set of parameters 
        - std_te : Array with the standard deviation of the test accuracy of each fold for each set of parameters 
        - idx_best : index of the best set of parameters
    ''' 
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Define lists to store the loss of training data and test data
    acc_tr = []
    std_tr = []
    acc_te = []
    std_te = []
    
    for params in parameters:
        acc_tr_, acc_te_ = cross_validation(y, x, k_indices, k_fold, method,log,  **params)
        acc_tr.append(np.mean(acc_tr_))
        std_tr.append(np.std(acc_tr_))
        acc_te.append(np.mean(acc_te_))
        std_te.append(np.std(acc_te_))
    
    idx_best =  np.argsort(-np.array(acc_te)) # Theminus sign here to get in descending order
        
    return acc_tr, acc_te, std_tr, std_te, idx_best