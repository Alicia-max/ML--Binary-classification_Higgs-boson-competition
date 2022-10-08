import numpy as np
from helpers import batch_iter


def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error. 
    inputs are the targeted y, the sample matrix tx and the feature vector w. 
    """
    e = y - tx.dot(w)
    mse = 1/2*np.mean(e**2)
    return mse

def compute_gradient(y, tx, w):

    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
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
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

    return losses, ws

def least_squares(y, tx):
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w= np.linalg.solve(a, b)
    loss= compute_mse(y, tx, w)
    
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

