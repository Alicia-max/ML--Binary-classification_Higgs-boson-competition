import numpy as np
from project.helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # compute loss, gradient
        grad, err = compute_gradient_mse(y, tx, w)
        loss = compute_mse(err)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
       

    return losses, ws
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_mse(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

    return losses, ws

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
  
    a = tx.T.dot(tx) + 2*tx.shape[0]*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w):
    '''
    
    '''
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    
    '''
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    
    '''
    pass

