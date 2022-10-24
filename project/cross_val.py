# -*- coding: utf-8 -*-

import numpy as np
from implementations import *
from preprocessing import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k_fold, method, log=False,  **params):
    '''
    TODO
    ''' 
    acc_tr_tmp=[]
    acc_te_tmp=[]
    
    degree = params['degree']
    offset = params['offset']
    
    params_without_degree_offset = params
  
    del params_without_degree_offset['degree']
    del params_without_degree_offset['offset']
   
    
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
            
            
            std_tx_tr = standardize(tx_tr)
            std_tx_te = standardize(tx_te)
           
            
            if(offset): 
                std_tx_tr = add_offset(std_tx_tr)
                std_tx_te = add_offset(std_tx_te)
            
            w, loss = method(y_tr, std_tx_tr, **params_without_degree_offset)
        

            #access accuracy
            if(log) : 
                acc_tr_tmp.append(accuracy(y_tr, predict_log(std_tx_tr,w)))
                acc_te_tmp.append(accuracy(y_te, predict_log(std_tx_te,w)))
            
            else : 
                acc_tr_tmp.append(accuracy(y_tr, predict(std_tx_tr,w)))
                acc_te_tmp.append(accuracy(y_te, predict(std_tx_te,w)))
       
    acc_tr=np.mean(acc_tr_tmp)
    acc_te=np.mean(acc_te_tmp)
            
    return acc_tr, acc_te

def cross_tunning(y, x, k_fold, method, parameters, seed, log = False) :
    '''
    TODO
    '''
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    acc_tr = []
    acc_te = []
    
    for params in parameters:
        acc_tr_, acc_te_ = cross_validation(y, x, k_indices, k_fold, method,log,  **params)
        acc_tr.append(acc_tr_)
        acc_te.append(acc_te_)    
    
    idx_best =  np.argmax(acc_te)      
        
    return acc_tr, acc_te,  idx_best