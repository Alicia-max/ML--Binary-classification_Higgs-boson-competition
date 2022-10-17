import numpy as np
from project.helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    TODO
    """
  
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # compute loss, gradient
        grad, err = compute_gradient_mse(y, tx, w)
        loss = compute_mse(err)
        # update w by gradient descent
        w = w - gamma * grad       

    return w, loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    TODO
    """
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
    Performs simple leat square regression

    Arguments:
        y: np.array of size [n]
            array of binary targets
        tx: np.array of size [n,d]
            array of d-dimensional features
    Returns:
        w: np.array of size [d]
            weights computed with the least squares method
        loss: scalar
            computed corresponding MSE loss
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

