import numpy as np
from helpers import *

# -*------------------------- Tools ---------------------------------*-


def _verify_range (x, limits) : 
    """
    Check the received array for extreme values defined by the threshold and return the modified array
    Input : 
        - x : array of floats
        - limits : array defining the upper and lower boundaries
    """
    x[x>limits[0]]=limits[0]
    x[x<limits[1]]=limits[1]
    return x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Inputs  
        - y : the output desired values
        - x : input data 
        - batch_size : ?
        - num_bacthes : 
        - Shuffle : Define if the data is randomly shuffled (avoid ordering in the original data 
        messing with the randomness of the minibatches) 
        
    Output 
        - iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`
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
    Compute and return the Mean Square Error. 
    Input 
        - y : targetd y 
        - x:  data Matrix
        - w : features vectors
    """
    
    #if the error is found out of the boundaries then the loss becomes infinite 
    #so we limite it bewteen 1e150& -1e150  (find in en emprical way)
    
    e = _verify_range(y - tx.dot(w), [1e150,-1e150])
    mse = (e**2/(2*len(e)))
    return mse

def compute_rmse(y, tx, w) : 
    """
    Compute and return the Root Mean Square Error. 
    Input 
        - y : targetd y 
        - x:  data Matrix
        - w : features vectors
        """
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)

def calculate_loss_log(y, tx, w):
    """
    A revoir 
    Compute and return the loss for a logistic Regression. 
    Input 
        - y : targetd y 
        - x:  data Matrix
        - w : features vectors
    """
    eta = tx@w
    #if eta >700 then loss become infinite, so we decide to limit it to 700 (found in an emprical fashion way)
    eta[eta > 700] = 700
    return np.sum(np.log(1+np.exp(eta))-y*(eta))

    

# -*------------------------- Gradient ---------------------------------*-

def compute_gradient_mse(y, tx, w):
    """
    Compute and return the gradient of the MSE loss 
    Input 
        - y : 
        - tx: 
        - w : 
    """
    err = _verify_range(y - tx.dot(w), [1e150,-1e150])
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    
    return grad


def sigmoid(t):
    """apply sigmoid function on t."""
    t[t<-700]=-700
    return (1.0 / (1 + np.exp(-t)))

# -*------------------------- Predication and Accuracy ---------------------------------*-

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

def predict_log(x, w): 
    """
    Compute the prediction of the model 
    Input : 
        - x : the feature Matrix
        - w : weights derived from the model
    """
    y_pred= sigmoid(np.matmul(x, w))
    y_pred[np.where(y_pred<.5)]=-1
    y_pred[np.where(y_pred>=.5)]=1
    return y_pred

# -*------------------------- METHODS---------------------------------*-

def least_squares_GD(y, tx, initial_w=None, max_iters=50, gamma=0.1):
    """
       fleeeeeme 
    Input : 
        - y : the predictor
        - tx : the feature Matrix
        - initial_w : 
        - max_iters : maximum number of iteration for GD 
        - gamma : step size for GD
    Ouput : 
        - w : computed weights 
        - loss : rmse loss
    """
    
    #initiate weights randomly
    if (initial_w is None) : initial_w = np.random.rand(tx.shape[1])
    
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient_mse(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad    
        
    loss=compute_rmse(y, tx, w) 
    
    return w, loss

def least_squares_SGD(y, tx, initial_w= None, max_iters=50, gamma=0.1):
    """
    fleeeeeme
     Input : 
        - y : the predictor
        - tx : the feature Matrix
        - initial_w : 
        - max_iters : maximum number of iteration for GD 
        - gamma : step size  GD
    Ouput : 
        - w : computed weights 
        - loss : rmse loss
    """
    
    #initiate weights randomly
    if (initial_w is None) : initial_w = np.random.rand(tx.shape[1])
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_mse(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            
    # calculate loss
    loss = compute_rmse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    fleeeeeme
      Input : 
        - y : the predictor
        - tx : the feature Matrix
        
    Ouput : 
        - w : computed weights 
        - loss : rmse loss
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    fleeeeeme
    Input : 
        - y : the predictor
        - tx : the feature Matrix
        - lambda_ :learning rate L1 regression
    Ouput : 
        - w : computed weights 
        - loss : rmse loss
    """
  
    a = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_rmse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-8):
    '''
    T  fleeeeeme
      Input : 
        - y : the predictor
        - tx : the feature Matrix
        - initial_w : 
        - max_iters : maximum number of iteration for GD 
        - gamma : step size GD 
    Ouput : 
        - w : computed weights 
        - loss : rmse loss
    '''
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)
    w = initial_w
    
    if initial_w is None:
        w = np.zeros(tx.shape[1])
        
    losses=[]
    for i in range(max_iters):
        
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        
        # compute loss  
        loss = calculate_loss_log(y, tx, w)
        losses.append(loss)
       
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return w, losses[-1]


def penalized_logistic_regression(y, tx, w, lambda_, gamma):
    """
      fleeeeeme
      Input : 
        - y : the predictor
        - tx : the feature Matrix
        - lambda_ : learning rate L1 regression
        - gamma : step size GD 
    Ouput : 
        - w : computed weights 
        - loss : mse loss
    """
    
    gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    w=w-gamma*gradient
    loss = calculate_loss_log(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    
    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-8):
    """
    fleeeeeme
      Input : 
        - y : the predictor
        - tx : the feature Matrix
        - lambda_ : learning rate L1 regression
        - gamma : step size GD 
    Ouput : 
        - w : computed weights 
        - loss : mse loss
    """
    # In case the targets are still in (-1, 1) range
    y = np.maximum(0, y)

    w = initial_w
    if initial_w is None:
        w = np.zeros(tx.shape[1])
        
    losses=[]

    for i in range(max_iters):
        
        w, loss = penalized_logistic_regression(y, tx, w, lambda_, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]

