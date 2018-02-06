# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan

"""

import numpy as np
import sys
from My_activations import *

##########################################
#### BACKWARD PROPAGATION FUNCTIONS  #####
##########################################

def activate_backward(dA, activation_cache):
    """
    Generic activation function
    Launches proper activation function according to 'activation' value from Layer_cache
    Allows the use of a specific activation function for hidden and output layers
    """
    Z, activation = activation_cache
     
    if activation == "sigmoid":
        dZ = dSigmoid(dA, Z)
        
    elif activation == "relu":
        dZ = dRelu(dA, Z)
        
    elif activation == "leaky_relu" :
        dZ = dLeaky_relu(dA, Z)
        
    elif activation == "tanh" :
        dZ = dTanH(dA, Z)
    
    elif activation == "identity":
        dZ = dIdentity(dA, Z)
    
    else:
         sys.exit(str(activation) + ' is not a valid activation function. Abort.') 
                  
    return dZ
    

def linear_backward(dZ, linear_cache):
    """
    Linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost (error) with respect to the linear output (of current layer l)
    linear_cache -- tuple of values (A_prev, W, b, Z, activation, D, keep_prob) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, Layer_cache):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache, KP_cache = Layer_cache
    
    dZ = activate_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward_propagation(Yhat, Y, caches):
    """
    Backward propagation for the [LINEAR->ACTIVATION] * (L-1) -> LINEAR -> OUTPUT group
    
    Arguments:
    X -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() for layer L (Yhat) (it's caches[L-1])
    
    Returns:
    grads -- A set with the gradients dA, dW, db
              
    """
    dA = []
    dW = []
    db = []
    L = len(caches) # the number of layers
    ## print("Caches of lenght:", L)
    ## Y = Y.reshape(Yhat.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation - calculate derivative of Yhat from output layer
    dYhat = - np.divide(Y, Yhat) + np.divide(1 - Y, 1 - Yhat)
    
    ## Get gradients for layer L-1 using dYhat from output layer                              
    Layer_cache = caches[L-1]
    ## print("Looking for gradients of layer:", L-1)
    dA_temp, dW_temp, db_temp = linear_activation_backward(dYhat, Layer_cache)
    ## insert layer gradients into gradients lists
    dA.append(dA_temp)
    dW.append(dW_temp)
    db.append(db_temp)
        
    for l in reversed(range(L-1)):
        ## Get the cache for layer l
        Layer_cache = caches[l]
        ## print("Looking for gradients of layer:", l)
        ## Get gradients for layer l using dA from previously inserted layer                           
        dA_temp, dW_temp, db_temp = linear_activation_backward(dA[0], Layer_cache)
        ## insert previous layer into gradients lists
        dA.insert(0,dA_temp)
        dW.insert(0,dW_temp)
        db.insert(0,db_temp)
        
    grads = (dA, dW, db)
    
    return grads


def L_model_backward_propagation_with_regularization(Yhat, Y, caches, lambd=0):
    """
    Backward propagation of baseline model with L2 regularization.
    
    Arguments:
    Yhat -- probability vector, output of the forward propagation (L_model_forward())
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    grads -- A set with the gradients dA, dW, db
    """
    dA = []
    dW = []
    db = []
    L = len(caches) # the number of layers
    m = Yhat.shape[1]
    assert(m > 0)
    
    Y = Y.reshape(Yhat.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation - calculate derivative of Yhat from output layer
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
    
    ## Get gradients for layer L-1 using dYhat from output layer                              
    Layer_cache = caches[L-1]                    ## That is the linear_cache (A, W, b), activation_cache (Z, activation) and KP_cache (D, Keep_prob) of layer L
    linear_cache, activation_cache, KP_cache = Layer_cache
    A, W, b = linear_cache      

    ## Get gradients for layer L using dYhat from output layer                              
    dA_temp, dW_temp, db_temp = linear_activation_backward(dYhat, Layer_cache)
    ## insert layer gradients into gradients lists
    dA.append(dA_temp)
    dW.append(dW_temp + ((lambd / m) * W))
    db.append(db_temp)
    
    for l in reversed(range(L-1)):
        ## Get the cache for layer l
        Layer_cache = caches[l]
        ## Unfold the layer cache as linear_cache (A, W, b), activation_cache (Z, activation) and KP_cache (D, Keep_prob) of layer l                                
        linear_cache, activation_cache, KP_cache = Layer_cache
        ## Unfold the linear_cache (A, W, b)
        A, W, b = linear_cache      
        ## Get gradients for layer l using dA from previously inserted layer                          
        dA_temp, dW_temp, db_temp = linear_activation_backward(dA[0], Layer_cache)
        ## insert previous layer into gradients lists
        dA.insert(0,dA_temp)
        dW.insert(0,dW_temp + ((lambd / m) * W))
        db.insert(0,db_temp)
 
    grads = (dA, dW, db)

    return grads


def L_model_backward_propagation_with_dropout(Yhat, Y, caches):
    """
    Backward propagation of baseline model with dropout.
    
    Arguments:
    Yhat -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    caches -- caches output from forward_propagation_with_dropout()
    
    Returns:
    gradients -- A list with the gradients with respect to each parameter, activation and pre-activation variables
    """
    dA = []
    dW = []
    db = []
    L = len(caches) # the number of layers
    
    ## m = Yhat.shape[1]
    Y = Y.reshape(Yhat.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation - calculate derivative of Yhat from output layer
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
    
    ## Get gradients for last layer L-1 using dYhat from output layer                              
    Layer_cache = caches[L-1]    ## That is the linear_cache (A, W, b) and activation_cache (Z) of layer L-1 

    ## Unfold the layer cache as linear_cache (A, W, b), activation_cache (Z, activation) and KP_cache (D, Keep_prob) of layer l                                
    linear_cache, activation_cache, KP_cache = Layer_cache
    
    ## Get gradients for layer L using dYhat from output layer                              
    dA_temp, dW_temp, db_temp = linear_activation_backward(dYhat, Layer_cache)
    ## insert layer gradients into gradients lists
    dA.append(dA_temp)
    dW.append(dW_temp)
    db.append(db_temp)
    
    ## No dropout allowed on last layer 
    ## Calculate gradients and apply dropout on all other hidden layers
    for l in reversed(range(L-1)):
        ## Get the cache for sublayer l
        Layer_cache = caches[l]      ## That is the linear_cache (A, W, b), activation_cache (Z, activation) and KP_cache (D, Keep_prob) of layer l

        ## Unfold the layer cache as linear_cache (A, W, b), activation_cache (Z, activation) and KP_cache (D, Keep_prob) of layer l                                
        linear_cache, activation_cache, KP_cache = Layer_cache

        ## Get gradients for layer l using dA from previously inserted layer                           
        dA_temp, dW_temp, db_temp = linear_activation_backward(dA[0], Layer_cache)

        ## Account for dropout use in forward propagation
        D, keep_prob = KP_cache
        dA_temp *= D
        dA_temp /= keep_prob
        
        ## insert gradients into gradients lists                
        dA.insert(0,dA_temp)
        dW.insert(0,dW_temp)
        db.insert(0,db_temp)

    grads = (dA, dW, db)
    
    return grads


###############################################
### UPDATE PARAMETERS FROM GRADIENT DESCENT ###
###############################################

def update_parameters(m, parameters, grads, learning_rate, lambd):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python list containing current layer weights: (W, b)
    grads -- python list containing output of L_model_backward, gradients for current layer: (dA, dW, db)
   
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    W, b = parameters
    dA, dW, db = grads
    L = len(W) # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(0, L-1):
        W[l] = W[l] - ((lambd / m) * W[l]) - (learning_rate * dW[l])
        b[l] = b[l] - (learning_rate * db[l])
        
    parameters = (W, b)
    
    return parameters


def update_parameters_with_momentum(m, parameters, grads, v, beta = 0.9, learning_rate = 0.01, lambd = 0):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python list containing current layer weights: (W, b)
    grads -- python list containing gradients for current layer: (dA, dW, db)
    v -- python list containing the current layer velocity tuple: (vW, vb)
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python list containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """
    W, b = parameters
    dA, dW, db = grads
    vW, vb = v
    L = len(W) # number of layers in the neural network
    
    # Momentum update for each parameter
    for l in range(0, L-1):
        
        # compute velocities
        vW[l] = (beta * vW[l]) + ((1 - beta) * dW[l])
        vb[l] = (beta * vb[l]) + ((1 - beta) * db[l])
        # update parameters
        W[l] = W[l] - ((lambd / m) * W[l]) - (learning_rate * vW[l])
        b[l] = b[l] - ((lambd / m) * b[l]) - (learning_rate * vb[l])
    
    parameters = (W, b)
    v = (vW, vb)
    
    return parameters, v


def update_parameters_with_adam(m, parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, lambd = 0):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python list containing current layer weights: (W, b)
    grads -- python list containing gradients for current layer: (dA, dW, db)
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python list containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    W, b = parameters
    dA, dW, db = grads
    vW, vb = v
    sW, sb = s

    L = len(W)                                # number of layers in the neural networks
    vW_corrected = []                         # Initializing first moment estimate, python list
    vb_corrected = []                         # Initializing first moment estimate, python list
    sW_corrected = []                         # Initializing second moment estimate, python list
    sb_corrected = []                         # Initializing second moment estimate, python list
    
    # Perform Adam update on all parameters
    for l in range(0, L-1):
        ## print("Adam for layer:",l)
        ## print("vW:",vW[l].shape,", vb:",vb[l].shape)
        ## print("sW:",sW[l].shape,", sb:",sb[l].shape)

        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        vW[l] = (beta1 * vW[l]) + ((1 - beta1) * dW[l])
        vb[l] = (beta1 * vb[l]) + ((1 - beta1) * db[l])

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        vW_corrected.append(vW[l] / (1 - np.power(beta1,t)))
        vb_corrected.append(vb[l] / (1 - np.power(beta1,t)))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        sW[l] = (beta2 * sW[l]) + ((1 - beta2) * np.square(dW[l]))
        sb[l] = (beta2 * sb[l]) + ((1 - beta2) * np.square(db[l]))

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        sW_corrected.append(sW[l] / (1 - np.power(beta2,t)))
        sb_corrected.append(sb[l] / (1 - np.power(beta2,t)))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        W[l] = W[l] - ((lambd / m) * W[l]) - (learning_rate * (vW_corrected[l] / np.sqrt(sW_corrected[l] + epsilon)))
        b[l] = b[l] - ((lambd / m) * b[l]) - (learning_rate * (vb_corrected[l] / np.sqrt(sb_corrected[l] + epsilon)))
        ### END CODE HERE ###

    parameters = (W, b)
    v = (vW, vb)
    s = (sW, sb)

    return parameters, v, s

