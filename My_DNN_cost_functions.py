# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan

"""

import numpy as np

#####################################
#####      LOSS  FUNCTIONS      #####
#####################################


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = np.sum(abs(y-yhat))
    
    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function
    """
    
    loss = np.sum(np.square(y-yhat))
    
    return loss


def Log_loss(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function
    """
    
    loss = np.sum((y*np.log(yhat)) + ((1-y)*np.log(1-yhat)))
    
    return loss

    
######################################
##### COMPUTING COST FUNCTION J  #####
######################################
    

def compute_cross_entropy_cost(Yhat, Y):
    """
    Implement the cross-entropy cost J(W,b) = SUM of Loss(Yhat, Y).

    Arguments:
    Yhat -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    J -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from Yhat and Y.
    J = -(1 / m) * Log_loss(Yhat, Y)
    J = np.squeeze(J)      # To make sure the cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(J.shape == ())
    
    return J


def compute_cost_with_regularization(Yhat, Y, parameters, lambd=0):
    """
    Implement the cost function with L2 regularization. 
    
    Arguments:
    Yhat -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function
    """
    m = Y.shape[1]
    cross_entropy_cost = 0
    L2_regularization_cost = 0
    
    ## retrieve W weights from parameters vector [W1, ..., WL] and [b1, ..., bL]
    W, b = parameters
    
    ## Calculate cross-entropy cost
    cross_entropy_cost = compute_cross_entropy_cost(Yhat, Y) # This gives you the cross-entropy part of the cost
    
    ## calculate L2 regularization factor
    L = len(W)
    sum_sqrs_W = 0
    for l in range(L-1):
        sum_sqrs_W += np.sum(np.square(W[l]))
    
    L2_regularization_cost = (1 / m) * (lambd / 2) * sum_sqrs_W
    
    ## Calculate overall cost
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
