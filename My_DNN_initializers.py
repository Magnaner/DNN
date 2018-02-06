# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan
Code inspired from Andrew Ng Stanford University
Deeplearning.ai Specialization courses on Coursera
"""
###########################
#####    L O A D      #####
###########################

import numpy as np

#####################################
##### INITIALIZE WEIGHTS & BIAS #####
#####################################


def initialize_parameters(layers_dims, init_type):
    # Initialize parameters dictionary.  "random", "he", "xavier", "bengio"
    if init_type == "random":
        parameters = initialize_parameters_random(layers_dims)
        
    elif init_type == "he":
        parameters = initialize_parameters_he(layers_dims)
        
    elif init_type == "xavier":
        parameters = initialize_parameters_tanh_Xavier(layers_dims)
        
    elif init_type == "bengio":
        parameters = initialize_parameters_tanh_Bengio(layers_dims)
        
    return parameters


def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    W = []
    b = []
    L = len(layers_dims)         # integer representing the number of layers
    
    for l in range(1, L):
        W.append(np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01)
        b.append(np.zeros((layers_dims[l], 1)))

    parameters = (W, b)
    
    return parameters


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    W = []
    b = []
    L = len(layers_dims)       # integer representing the number of layers
    
    for l in range(1, L):
        W.append(np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1]))
        b.append(np.zeros((layers_dims[l], 1)))

    parameters = (W, b)
       
    return parameters


def initialize_parameters_tanh_Xavier(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    W = []
    b = []
    L = len(layers_dims) # integer representing the number of layers
     
    for l in range(1, L):
        W.append(np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(1./layers_dims[l-1]))
        b.append(np.zeros((layers_dims[l], 1)))
        
    parameters = (W, b)
       
    return parameters


def initialize_parameters_tanh_Bengio(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python tuple containing arrays W[l] and b[l]:
                    W[1] -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b[1] -- bias vector of shape (layers_dims[1], 1)
                    ...
                    W[L] -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    b[L] -- bias vector of shape (layers_dims[L], 1)
    """
    
    W = []
    b = []
    L = len(layers_dims)  # integer representing the number of layers
     
    for l in range(1, L):
        W.append(np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./(layers_dims[l-1] + layers_dims[l])))
        b.append(np.zeros((layers_dims[l], 1)))
        
    parameters = (W, b)
       
    return parameters


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python tuple containing arrays W[l] and b[l].
    
    Returns: 
    vW, vb -- python vectors that will contain the exponentially weighted average of the gradient.
                    vW[l] = dW[l]...
                    vb[l] = db[l]...
    sW, sb -- python vectors that will contain the exponentially weighted average of the squared gradient.

    """
    W, b = parameters
    L = len(W)  # number of layers in the neural networks
    vW = []
    vb = []
    sW = []
    sb = []
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(0,L-1):
        vW.append(np.zeros((W[l].shape)))
        vb.append(np.zeros((b[l].shape)))
        sW.append(vW[l])
        sb.append(vb[l])
    
    v = (vW, vb)
    s = (sW, sb)
    return v, s
