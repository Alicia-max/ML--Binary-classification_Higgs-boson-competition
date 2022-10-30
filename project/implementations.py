# -*- coding: utf-8 -*-
import numpy as np
from helpers import *

# -*------------------------- Tools ---------------------------------*-

def _verify_range (x, limits) : 
    """
    Verifies the received array for extreme values defined by the threshold and return the modified array
    Input : 
        - x : array of floats
        - limits : array defining the upper and lower boundaries
    """
    x[x>limits[0]]=limits[0]
    x[x<limits[1]]=limits[1]
    return x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generates a minibatch iterator for a dataset.
    Inputs  
        - y : Expected value vector
        - x : Data Matrix
        - batch_size : Number of data points sample (one)
        - num_bacthes : Number of batch (one)
        - Shuffle : Define if the data is randomly shuffled (avoid ordering in the original data 
        messing with the randomness of the minibatches) 
        
    Output 
        - iterators which gives mini-batches of `batch_size` matching elements from `y` and `tx`
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

# -*------------------------- LOSS ---------------------------------*-

def compute_mse(y, tx, w):
    """
    Computes and returns the Mean Square Error. 
    Input 
        - y : Expected value vector
        - x:  Data Matrix
        - w : Weight Vector
    """
    
    # If the error is found out of the boundaries then the loss becomes infinite 
    # So we limite it bewteen 1e150& -1e150  (find in en emprical way)  
    e = _verify_range(y - tx.dot(w), [1e150,-1e150])
    
    # Factor of 0.5 to be consistent with the course
    mse = (e**2/(2*len(e)))
    return mse

def compute_rmse(y, tx, w) : 
    """
    Computes and Returns the Root Mean Square Error. 
    Input 
        - y : Expected value vector
        - x:  Data Matrix
        - w : Weight vector
        """
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)

def calculate_loss_log(y, tx, w):
    """ 
    Computes and Returns the Logistic loss  
    Input 
        - y : Expected Value vector
        - x:  Data Matrix
        - w : Weight vector
    """
    eta = (np.matmul(tx, w))
    eta[eta > 700] = 700
    return (1/len(y))*( np.sum(np.log(1+np.exp(eta))-y*(eta)))

# -*------------------------- Gradient ---------------------------------*-

def compute_gradient_mse(y, tx, w):
    """
    Computes and Returns the gradient of the loss with respect to the features vector (w)
    Input 
        - y : Expected value vector
        - tx: Data Matrix
        - w : Weight vector
    """
    err = _verify_range(y - tx.dot(w), [1e150,-1e150])
    grad = -tx.T.dot(err) / len(err)
    return grad

def compute_gradient_logistic(y, tx, w):
    
    """
    Computes and Returns the gradient of the loss with respect to the features vector (w)
    Input 
        - y : Expected value vector
        - tx: Data Matrix
        - w : Weight vector
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def sigmoid(t):
    """Apply sigmoid function on the input parameter t."""

    #limit the value of t to avoid the infinite term
    t[t<-700]=-700
    return (1.0 / (1 + np.exp(-t)))

# -*------------------------- Predication and Accuracy ---------------------------------*
def accuracy(y,y_pred):
    """
    Computes the accuracy as the average of the correct predictions and returns it
    Input : 
        - y : the true prediction 
        - y_pred : model's prediction
    """
    score=0
    for idx, val in enumerate(y): 
        if val==y_pred[idx] : 
            score+=1
            
    return score/len(y)

def predict(x, w): 
    """
    Compute and return the model's Prediction for a Linear regression
    Input : 
        - x : Data Matrix
        - w : Weight vector
    """
    # Compute the prediction
    y_pred=(np.matmul(x, w))
    
    # Assign the class of the prediction according to the sign
    y_pred[np.where(y_pred<0)]=-1
    y_pred[np.where(y_pred>=0)]=1
    return y_pred

def predict_log(x, w): 
    """
    Compute the and return model's prediction for a Logistic regression
    Input : 
        - x : Data Matrix
        - w : Weight vector
    """
    # Compute the Prediction
    y_pred= sigmoid(np.matmul(x, w))
    
    # Assign the class of the prediction according to the position around 0.5 
    # (the class becomes 0 or 1 for the Logistics case)
    y_pred[np.where(y_pred<.5)]=-1
    y_pred[np.where(y_pred>=.5)]=1
    return y_pred

# -*------------------------- METHODS---------------------------------*-

def mean_squared_error_gd(y, tx, initial_w=None, max_iters=50, gamma=0.1):
    """
    Computes the weights and associated loss for a linear regression solving y=tx@w using Gradient Descent.
    The gradient descent algorithm aims to find the optimal weights by minimizing the MSE (||y-tx@w|^2).
    (While gradient descent can provide a local minimum, the least squares method converges to the global minimum.)
    
    Input : 
        - y : Expected value
        - x : Data Matrix
        - initial_w : Initial weight vector (set to 0 if not given) to start the GD algorithm.
        - max_iters : Number of iteration for GD
        - gamma : learning rate for GD 
    Output : 
        - w : Weight Vectors
        - loss : RMSE of the calculated prediction (tx@w) and the expected value y using the final w 
    """
  
    # Initiate weights to zero since they tend to be small during optimization
    if (initial_w is None) : initial_w = np.zeros(tx.shape[1])
    w = initial_w
    
    # Start the Linear Regression
    for n_iter in range(max_iters):
        # Compute gradient & update w
        grad = compute_gradient_mse(y, tx, w)
        w = w - gamma * grad    

    loss=compute_rmse(y, tx, w) 
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w= None, max_iters=50, gamma=0.1):
    """
    Computes the weights and associated loss for a linear regression solving y=tx@w using stochastic gradient descent.
    Unlike GD, SDG uses only one training sample to compute the gradients and thus find the optimal weight.
 
    Input : 
        - y : Expected value
        - x : Data Matrix
        - initial_w : Initial weight vector (set to 0 if not given) to start the GD algorithm.
        - max_iters : Number of iteration for GD
        - gamma : learning rate for GD 
    Output : 
        - w : Weight Vectors
        - loss : RMSE of the calculated prediction (tx@w) and the expected value y using the final w 
    """
    
    # Initiate weights to zero since they tend to be small during optimization
    if (initial_w is None) : initial_w = np.zeros(tx.shape[1])
    w = initial_w
    
    # Start Linear Regression
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # Compute a stochastic gradient & update w
            grad = compute_gradient_mse(y_batch, tx_batch, w)
            w = w - gamma * grad
      
    # Calculate last loss
    loss = compute_rmse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    Computes a closed-form solution of the problem y = tx @ w, and the associated error.
    This method aims at finding w such that it minimizes the MSE ||y-tx@w||^2 (global optimum). 
    Input : 
        - y : Expected value vector
        - x : Data Matrix
       
    Output : 
        - w : Weight Vectors
        - loss : RMSE of the calculated prediction (tx@w) and the expected value y
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    loss = compute_rmse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Least square Regression with a regularization term lambda_. 
    This method aims at finding w such that it minimizes the ||y-tx@w||^2 + lambda_*||w||^2.
    Input : 
        - y : Expected value vector
        - tx : Data Matrix
        - lambda_ :  L1- Regularization term 
    Ouput : 
        - w : Weight Vectors 
        - loss : RMSE of the computed prediction (tx@w) and the expected value y
    """
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    loss = compute_rmse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-8):
    '''
    Computes the weights and associated loss for a logistic regression solving y=sigmoid(tx @ w) using Gradient Descent. 
    This method aims at finding w such that it it minimizes the negative log likelihood (may find local mimimum).

      Input : 
        - y : Expected value error
        - tx : Data Matrix
        - initial_w : Initial weight vector (set to 0 if not given) to start the GD algorithm.
        - max_iters : Number of iteration for GD
        - gamma : learning rate for GD 
        - threshold : defined how much the loss has to changed over the iteration to continue the GD Algorithm
        
    Ouput : 
        - w : Weight Vectors
        - loss : RMSE of the computed prediction and the expected value y
    '''
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)
    w = initial_w
    
    # Initiate weights to zero since they tend to be small during optimization
    if (initial_w is None) : initial_w = np.zeros(tx.shape[1])
    w = initial_w
    losses=[]
    
    # Start Logistic Regression
    for i in range(max_iters):
    
        # Compute gradient & update w
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        
        # Compute loss  
        loss = calculate_loss_log(y, tx, w)
        losses.append(loss)
        
        # Stop the search if the loss has not changed enough
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w, losses[-1]

def penalized_logistic_regression(y, tx, w, lambda_, gamma):
    """
    Computes one step of gradient descent for regularized logistic regression.
    Input : 
        - y : Expected value vector
        - x : Data Matrix
        - lambda_ : L1- Regularization term 
        - gamma : learning rate GD 
    output : 
        - w : Weight vectors
        - loss : RMSE of the prediction and the expected value 
    """
    
    gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    w=w-gamma*gradient
    loss = calculate_loss_log(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-8):
    '''
    Computes the weights and associated loss for a logistic regression solving y=sigmoid(tx @ w)
    using Gradient Descent and using a penalization term lambda_.
    This method aims at finding w such that it it minimizes the negative log likelihood (may find local mimimum).

      Input : 
        - y : Expected value error
        - tx : Data Matrix
        - lambda_ : learning rate L1 Regression
        - gamma : learning rate for GD 
        - threshold : defined how much the loss has to changed over the iteration to continue the GD Algorithm
        
    Ouput : 
        - w : Weight Vectors
        - loss : RMSE of the computed prediction and the expected value y
    '''
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)

    # Initiate weights to zero since they tend to be small during optimization
    if (initial_w is None) : initial_w = np.zeros(tx.shape[1])
    w = initial_w
    losses=[]
    
    # Start Logistic Regression
    for i in range(max_iters):
        # Compute gradient & update w
        w, loss = penalized_logistic_regression(y, tx, w, lambda_, gamma)
        losses.append(loss)
        
        # Verify if the loss has sufficiently changed
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
