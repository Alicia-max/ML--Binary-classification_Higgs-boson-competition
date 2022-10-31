# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from preprocessing import *


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold
    """
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
        - k_indices : The indices of the validation set k_indices
        - method : One of the 6 implemented method 
        - log : Boolean that determine the prediction used (predict or predict_log)
        - Method's parameters
    Output : 
        - acc_tr_tmp : Array with train accuracy  for each fold
        - acc_val_tmp : Array with validation accuracy for each fold
    ''' 
    
    acc_tr_tmp=[]
    acc_val_tmp=[]
    degree = params['degree']
    cross = params['cross']

    params_without_degree_cross = params
    del params_without_degree_cross['degree']
    del params_without_degree_cross['cross']
    
    for k in range(k_fold) :
        
        # Get k'th subgroup in validation, others in train
        val_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        
        y_val = y[val_indice]
        y_tr = y[tr_indice] 
        x_val = x[val_indice,:]
        x_tr = x[tr_indice,:]
        
        # Form data with polynomial degree
        tx_tr = build_poly(x_tr, degree)
        tx_val = build_poly(x_val, degree)
        
        
        # Cross-terms addition
        if cross:
            cross_terms_tr = cross_terms(x_tr)
            cross_terms_val = cross_terms(x_val)
                
            tx_tr = np.c_[tx_tr, cross_terms_tr]
            tx_val = np.c_[tx_val, cross_terms_val]
            
        # Standarization
        std_tx_tr, mean_tx, std_tx = standardize(tx_tr)
        std_tx_val, _, _= standardize(tx_val,mean_tx, std_tx)
        
        # Offset additions
        std_tx_tr = add_offset(std_tx_tr)
        std_tx_val = add_offset(std_tx_val)
     
        w, loss = method(y_tr, std_tx_tr, **params_without_degree_cross)
        
        # Access accuracy
        if(log) : 
            acc_tr_tmp.append(accuracy(y_tr, predict_log(std_tx_tr,w)))
            acc_val_tmp.append(accuracy(y_val, predict_log(std_tx_val,w)))
            
        else : 
            acc_tr_tmp.append(accuracy(y_tr, predict(std_tx_tr,w)))
            acc_val_tmp.append(accuracy(y_val, predict(std_tx_val,w)))
            
    return acc_tr_tmp, acc_val_tmp

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
        - acc_tr_tmp : Array with each train accuracy of each fold for each set of parameters 
        - acc_val_tmp : Array with each mean validation accuracy of each fold for each set of parameters 
        - idx_best : indexes of the best param list
       
    ''' 
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    acc_tr=[]
    acc_val=[]
    for params in parameters:
        acc_tr_, acc_val_ = cross_validation(y, x, k_indices, k_fold, method,log,  **params)
        acc_tr.append(acc_tr_)
        acc_val.append(acc_val_)
   
    idx_best =  np.argsort(-np.mean(acc_val, axis=1)) # The minus sign here to get in descending order
        
    return acc_tr, acc_val, idx_best