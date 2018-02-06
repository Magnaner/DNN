# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan

"""

import numpy as np
import sys
from My_activations import *

######################################
##### LINEAR FORWARD PROPAGATION #####
######################################
    
## GENERIC FORWARD PROP ACTIVATION CALLS
## Allows for selection of hidden layers and output layer activation functions through hyperparameters

def activate_forward(Z, activation="relu"):
    """
    Compute an activation function of z

    Arguments:
    Z -- A scalar or numpy array of any size.
    activation -- g[l] the activation to be applied to z in this layer, as string : "sigmoid", "relu", "leaky_relu", "tanh" or "identity"

    Return:
    A -- Results from the activation function g[l] on z : g[l](z)
    cache -- the input value Z and the activation function used
    """
    
    if activation == "sigmoid" :
        A = Sigmoid(Z)
        
    elif activation == "relu" :
        A = Relu(Z)
        
    elif activation == "leaky_relu" :
        A = Leaky_relu(Z)
        
    elif activation == "tanh" :
        A = TanH(Z)
        
    elif activation == "identity" :
        A = Identity(Z)
        
    else:
        sys.exit(str(activation) + ' is not a valid activation function. Abort.') 

    activation_cache = (Z, activation)
    
    return A, activation_cache                ## this is the activation_cache (Z, activation)

    
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation (y = mx + b).

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    linear_cache = (A, W, b)
   
    return Z, linear_cache             ## this is the linear_cache (A, W, b)


def linear_activation_forward(A_prev, W, b, activation="sigmoid"):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer a(l) = g[l](z[l])

    Arguments:
    A_prev -- activations from previous layer (a(l-1) or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- g[l] the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "leaky_relu", "tanh" or "identity"

    Returns:
    A -- a(l) the output of the activation function g[l](z[l]), also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activate_forward(Z, activation)
     
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A, linear_cache, activation_cache             ## this is the linear_cache and activation_cache (A, W, b, Z, activation)


def L_model_forward_propagation(X, parameters, acthid="relu", actout="sigmoid", keep_prob=1):
    """
    Implement forward propagation for the [LINEAR->ACTIVATION[h]]*(L-1)->LINEAR->ACTIVATION[AL] computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    acthid -- activation function for hidden layers ("sigmoid", "relu", "leaky_relu", "tanh" or "identity")
    actout -- activation function for the ouput layer ("sigmoid", "relu", "leaky_relu", "tanh" or "identity")
    keep_prob -- probability of keeping a neuron active during drop-out, scalar
              -- for the optional use of dropout. keep_prob = 1 means "dropout off"

    Returns:
    YhatL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    KP_cache = []
    kp = keep_prob
    YhatL = X
    W, b = parameters
    L = len(W)                  # number of layers in the neural network

    # Implement [LINEAR -> ACTIVATION[h]]*(L-1). Add "cache" to the "caches" list.
    for l in range(0, L-1):
        cache = []
        A_prev = YhatL
        D = 0
            
        ## Perform forward propagation on layer L
        YhatL, linear_cache, activation_cache = linear_activation_forward(A_prev, W[l], b[l], acthid)

        ## apply dropout if used
        if kp < 1 :
            D = np.random.rand(YhatL.shape[0], YhatL.shape[1])   # Initialize matrix D = np.random.rand(..., ...)
            D = (D < kp)                                         # Convert entries of D to 0 or 1 (using keep_prob as the threshold)
            YhatL = YhatL * D                                    # Shut down some neurons of A
            YhatL = YhatL / kp                                   # Scale the value of neurons that haven't been shut down
        KP_cache = (D, kp)

        ## now append linear and KP caches to caches for backprop
        cache = (linear_cache, activation_cache, KP_cache)
        caches.append(cache)      # cache of layer [l] is [A, W, b, Z, activation, D, keep_prob])

    ## Perform forward propagation on last layer using actout activation function, never dropout
    A_prev = YhatL
    KP_cache = (1,1)
    YhatL, linear_cache, activation_cache = linear_activation_forward(A_prev, W[L-1], b[L-1], actout)
    ## now append linear and KP caches to caches for backprop
    cache = (linear_cache, activation_cache, KP_cache)
    caches.append(cache)      # cache of layer [l] is [A, W, b, Z, activation, D, keep_prob])

    return YhatL, caches
