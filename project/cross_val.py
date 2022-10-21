# -*- coding: utf-8 -*-

import numpy as np
from implementations import *

def add_offset(x): 
    '''
    Add a column of 1 to the feature matrix x
    '''
    return (np.c_[np.ones(x.shape[0]), x])


def standardize(x):
    '''
    Standarize the data according the mean and the standard deviation
    and take as an input the features matrix X
    '''
    centered_data = x - np.mean(x, axis=0)
    std=np.std(centered_data, axis=0)
    
    std_data = centered_data[:,std>0] /std[std>0]
    
    std_0 = std[std <=0]
    
    if(len(std_0)>0) :
        raise ValueError("DIVISION BY 0 : There are features with standard deviation equal to 0")
 
    
    return std_data

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x),0))
    
   
    for deg in range(1, degree+1):
      
        poly = np.c_[poly, np.power(x, deg)]   
  
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold, method,  **params):
    
    acc_tr_tmp=[]
    acc_te_tmp=[]
    degree = params['degree']
    
   
    params_without_degree = params
    del params_without_degree['degree']
    for k in range(k_fold) :
          # get k'th subgroup in test, others in train
            te_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)
            
            y_te = y[te_indice]
            y_tr = y[tr_indice] 
            x_te = x[te_indice,:]
            x_tr = x[tr_indice,:]
            
            # form data with polynomial degree
           
            tx_tr = build_poly(x_tr, degree)
            tx_te = build_poly(x_te, degree)
           
            tx_tr_std=standardize(tx_tr)
            tx_te_std=standardize(tx_te)
         
            
        
            tx_tr_std = add_offset(tx_tr_std)
            tx_te_std = add_offset(tx_te_std)
            
            
            w, loss = method(y_tr, tx_tr_std, **params_without_degree)

            #access accuracy
            acc_tr_tmp.append(accuracy(y_tr, predict(tx_tr_std,w)))
            acc_te_tmp.append(accuracy(y_te, predict(tx_te_std,w)))
       
    acc_tr=np.mean(acc_tr_tmp)
    acc_te=np.mean(acc_te_tmp)
            
    return acc_tr, acc_te

def cross_tunning(y, x, k_fold, method, parameters, seed) :
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    acc_tr = []
    acc_te = []
    
    for params in parameters:
        acc_tr_, acc_te_ = cross_validation(y, x, k_indices, k_fold, method, **params)
        acc_tr.append(acc_tr_)
        acc_te.append(acc_te_)    
    
    idx_best =  np.argmax(acc_te)      
        
    return acc_tr, acc_te, idx_best
 