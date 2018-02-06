# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan
Code inspired from Andrew Ng Stanford University
Deeplearning.ai Specialization courses on Coursera
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from My_activations import *
from My_DNN_initializers import *
from My_DNN_Forward_prop import *
from My_DNN_cost_functions import *
from My_DNN_Backward_prop import *

#################################################
#################################################
#####  MAIN DEEP NEURAL NET TRAINING MODEL  #####
#################################################
#################################################

def set_default_hyperparameters():
    """
    Proposed good values :
        learning_rate = 0.0007, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, n_epochs = 10000, print_cost = True
    """
    ## BASIC HYPERPARAMETERS
    n_epoch = 3000
    print_cost = True
    mini_batch_size = 300
    init_type = "he"      ## "random", "he", "xavier", "bengio"
    acthid = "relu"       ## relu, tanh, leaky_relu, sigmoid 
    actout = "sigmoid"    ## sigmoid, softmax, tanh
    Lr = 0.001
    ## OPTIMNIZERS
    optimizer = "adam"    ## "gd", "adam", "momentum"
    beta1 = 0.9           ## 1 = off
    beta2 = 0.999         ## 1 = off
    epsilon = 1e-8
    ## REGULARIZATION METHODS
    lambd = 0.5           ## 0 = off
    keep_prob = 1.        ## 1 = off
    
    hyperparameters = (n_epoch, print_cost, mini_batch_size, acthid, actout, optimizer, Lr, beta1, beta2, epsilon, lambd, keep_prob, init_type)
    
    return hyperparameters


def show_hyperparameters(hyperparameters):
    ## unfold hyperparameters values
    n_epoch, print_cost, mini_batch_size, acthid, actout, optimizer, Lr, beta1, beta2, epsilon, lambd, keep_prob, init_type = hyperparameters
    
    ## PRINT hyperparameters
    print("BASIC HYPERPARAMETERS")
    print("=====================")
    print('Number of epoch :', n_epoch)
    print('Print cost: ',print_cost)
    print('Mini_batch_size: ',mini_batch_size)
    print('Learning rate: ', Lr)
    print('Hidden layers activation function: ', acthid)
    print('Output layer activation function: ', actout)
    print('Initialization type: ', init_type)
    print()
    print("OPTIMIZING HYPERPARAMETERS")
    print("==========================")
    print('Optimizer: ', optimizer)
    print('Beta1: ', beta1)
    print('Beta2: ', beta2)
    print('Epsilon: ', epsilon)
    print()
    print("REGULARIZATION HYPERPARAMETERS")
    print("==============================")
    print('Lambda: ', lambd)
    print('Dropout is ', str(keep_prob != 1))
    print()


def make_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make "random" minibatches the same using a seed
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k * mini_batch_size) : ((k+1) * mini_batch_size)]
        mini_batch_Y = shuffled_Y[:, (k * mini_batch_size) : ((k+1) * mini_batch_size)] 
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Step 3: Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (num_complete_minibatches * mini_batch_size) : m]
        mini_batch_Y = shuffled_Y[:, (num_complete_minibatches * mini_batch_size) : m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
   
    return mini_batches


def Train_L_layers_DNN_model(X, Y, layers_dims, hyperparams):
    """
    Implements an L-layers neural network: LINEAR->ACTIVATION->...->LINEAR->OUTPUT.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (output size, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    hyperparams -- list of hyperparameters in order num_iterations, print_cost, learning_rate, beta1, beta2, epsilon, lambd, keep_prob, initialization
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    ## initialize internal variables
    seed = 10                        # So that "random" minibatches are the same from run to run
    np.random.seed(seed)
    
    t = 0           ## initializing the counter required for Adam update
    grads = []      ## to gather a dictionary of gradients from backprop
    costs = []      ## to keep track of the loss
    m = X.shape[1]  ## number of training samples
    
    ## Get hyperparameters
    n_epoch, print_cost, mini_batch_size, acthid, actout, optimizer, learning_rate, beta1, beta2, epsilon, lambd, keep_prob, initialization = hyperparams
    show_hyperparameters(hyperparams)
    
    # Initialize weights, bias and gradients in parameters list
    parameters = initialize_parameters(layers_dims, initialization)    
    
    # Initialize the momentum or adam optimizer
    v, s = initialize_adam(parameters)
    
    # Loop (gradient descent)
    for i in range(0, n_epoch):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = make_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            Yhat, caches = L_model_forward_propagation(minibatch_X, parameters, acthid, actout, keep_prob=1)
        
            # Compute cost
            if lambd == 0 :
                cost = compute_cross_entropy_cost(Yhat, minibatch_Y)
            else :
                cost = compute_cost_with_regularization(Yhat, minibatch_Y, parameters, lambd)
                
            # Backward propagation
            if lambd == 0 and keep_prob == 1:
                grads = L_model_backward_propagation(Yhat, minibatch_Y, caches)
            elif lambd != 0:
                grads = L_model_backward_propagation_with_regularization(Yhat, minibatch_Y, caches, lambd)
            elif keep_prob < 1:
                grads = L_model_backward_propagation_with_dropout(Yhat, minibatch_Y, caches)
 
            # Update parameters
            ## parameters = update_parameters(m, parameters, grads, v, s, t, beta1, beta2, epsilon, lambd)
            if optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(m, parameters, grads, v, beta1, learning_rate, lambd)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(m, parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon, lambd)
            else:
                parameters = update_parameters(m, parameters, grads, learning_rate, lambd)
               
        # Print the cost every 100 training example
        ## print(info_param(parameters))
        ## print(info_grads(grads))
        ## print(info_caches(caches, parameters))
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    ## difference = gradient_check(parameters, grads, X, Y, epsilon)
    ## print(difference)
    
    return parameters


def predict(X, parameters, acthid, actout, Bool=True):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    Yhat, cache = L_model_forward_propagation(X, parameters, acthid, actout)
    if Bool:
        if actout == "sigmoid":
            p = np.where(Yhat >= 0.5, 1., 0.)
        else:
            p = np.where(Yhat >= 0, 1., 0.) 
        return p
    else:
        return Yhat
