# -*- coding: utf-8 -*-

import numpy as np
from implementations import *

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
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

def cross_validation(y, x, k_indices, k_fold, method, lambda_=None, gamma=None, degree=1, **kwargs):
    
    acc_tr_tmp=[]
    acc_te_tmp=[]
  
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
            
            # compute weights
            #without GD or SDG 
            if (gamma==None) : 
                if(lambda_==None) : 
                    w, loss = method(y_tr, tx_tr, **kwargs)
                    
                else: 
                     w, loss = method(y_tr, tx_tr, lambda_=lambda_, **kwargs)
                
            #With GD or SDG 
            else :
                if(lambda_==None) : 
                    w, loss = method(y_tr, tx_tr, gamma=gamma, **kwargs)    
                else  :
                    w, loss = method(y_tr, tx_tr, gamma=gamma, lambda_=lambda_, **kwargs)
                    
            #access accuracy
            acc_tr_tmp.append(accuracy(y_tr, predict(tx_tr,w)))
            acc_te_tmp.append(accuracy(y_te, predict(tx_te,w)))
       
    acc_tr=np.mean(acc_tr_tmp)
    acc_te=np.mean(acc_te_tmp)
            
    return acc_tr, acc_te

def cross_tunning(y, x, k_fold, method, params, name_param, seed, **kwargs) :
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    acc_tr = []
    acc_te = []
    
    # cross validation over the given params
    for param in params:
        if (name_param=='degree'):
            acc_tr_, acc_te_ = cross_validation(y, x, k_indices, k_fold, method, degree=param, **kwargs)     
        elif (name_param=='learning_rate'):
            acc_tr_, acc_te_= cross_validation(y, x, k_indices, k_fold, method, gamma=param, **kwargs)       
        
        acc_tr.append(acc_tr_)
        acc_te.append(acc_te_)
        
    ind_best_param =  np.argmax(acc_te)      
        
    return params[ind_best_param], acc_tr, acc_te
 