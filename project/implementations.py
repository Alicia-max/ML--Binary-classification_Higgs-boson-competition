import numpy as np
from helpers import *


# -*------------------------- LOSS & Stuff---------------------------------*-

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

def accuracy(y,y_pred):
    """
    Compute and return the accuracy
    Input : 
        - y : the true prediction 
        - y_pred : prediction from the model
    """
    score=0
    for idx, val in enumerate(y): 
        if val==y_pred[idx] : 
            score+=1
            
    return score/len(y)

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

def predict(x, w): 
    """
    Compute the prediction of the model 
    Input : 
        - x : the feature Matrix
        - w : weights derived from the model
    """
    y_pred=(np.matmul(x, w))
    y_pred[np.where(y_pred<0)]=-1
    y_pred[np.where(y_pred>=0)]=1
    return y_pred

# -*------------------------- METHODS---------------------------------*-

def least_squares_GD(y, tx, initial_w=0, max_iters=50, gamma=0.1):
    """
    flemme :)))
    Input : 
        - y : the predictor
        - tx : the feature Matrix
        - initial_w : 
        - max_iters : maximum number of iteration for GD 
        - gamma : learning rate of GD
    Ouput : 
        - w : computed weights 
        - loss : mse loss
    """
    
    #initiate weights randomly
    if (initial_w==0) : initial_w = np.random.rand(tx.shape[1])
    
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient_mse(y, tx, w)
        loss = compute_mse(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad       
        
    return w, loss
    
def least_squares_SGD(y, tx, initial_w= 0, max_iters=50, gamma=0.1):
    """
    
     Input : 
        - y : the predictor
        - tx : the feature Matrix
        - initial_w : 
        - max_iters : maximum number of iteration for GD 
        - gamma : learning rate of GD
    Ouput : 
        - w : computed weights 
        - loss : mse loss
    """
    
    #initiate weights randomly
    if (initial_w==0) : initial_w = np.random.rand(tx.shape[1])
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_mse(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    TODO
    """
  
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    TODO
    '''
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)

    w = initial_w
    loss = 0

    for i in range(max_iters):
        grad, _ = compute_gradient_logistic(y, tx, w)
        w = w - gamma*grad

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    TODO
    '''
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)

    w = initial_w
    loss = 0

    for i in range(max_iters):
        grad, _ = compute_gradient_logistic(y, tx, w)
        grad = grad + lambda_*w**2
        w = w - gamma*grad

    return w, loss

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
            