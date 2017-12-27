# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan
"""
###########################
#####    L O A D      #####
###########################
"""
# INITIAL IMPORT
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


#####################################
#####    U T I L I T I E S      #####
#####################################
'''
def parameters_to_vectors(dictionary):
    """
    Simple conversion from labeled data set to a scalar vector.
    
    Arguments:
        dictionary -- dictionary of parameters {Wx, Bx} of lenght L
        
    Returns:
        new_vector -- content of input dictionary values converted into a scalar vector (L*n*m,1)
        dictionary_structure -- data set of structure of input dictionary in the form {'key':(n,m)} of lenght L
        
    """
    ## Find how many layers in DNN
    ## Each layer has n neurons with {Wn, bn} or {dWn, dbn}, therefore dictionary is 2 X number of layers
    L = len(dictionary) // 2
    
    ## initialize internal variables
    new_vector = []                          ## to store en result vector
    dictionary_structure = {}                ## to store original dictionary structure
    dictionary_keys = list(dictionary)       ## to get a list of the the KEYS of original dictionary
                                             ## either ('Wn', 'bn') for parameters

    ## check if dictionary is empty
    if L == 0:
        sys.exit('*** ERROR : empty dictionary ***')
    
    for key in dictionary_keys :
        temp_vector = np.reshape(dictionary[key], (-1,1))       ## flattens a matix (n, m) to a vector of dim (n*m,1)
        dictionary_structure[key] = dictionary[key].shape       ## stores the original dimensions (n,m) for key
    
    for 1 in range(0,L):  
        W_vector = dictionary[W]
        gradientskey = 'dW'+str(L)
            if key in ['W1', gradientskey]:
                new_vector = temp_vector     ## necessary for first KEY since concatenate requires non-empty variables
            else :
                new_vector = np.concatenate((new_vector, temp_vector), axis=0)  ## add vector for key to new_vector
   
    return new_vector, dictionary_structure
                
  
def vector_to_dictionary(vector, dictionary_structure):
    """
    Simple conversion from scalar vector into original dictionary with original dimensions
    
    Arguments:
        vector, dictionary_structure -- output of dictionary_to_vector()
        
    Returns:
        new_dict -- content of vector reshaped into dictionary as per dictionary_structure
        
    """
    
    ## Find how many layers in DNN
    ## Each layer has n neurons with {Wn, bn} or {dWn, dbn}, therefore dictionary needs to have 2 X number of layers
    L = len(dictionary_structure) // 2

    ## initialize internal variables
    new_dict = {}                                  ## to restore original dictionary structure
    dictionary_keys = list(dictionary_structure)   ## to get a list of the the KEYS of original dictionary
                                                   ## either ('Wn', 'bn') for parameters or ('dWn', 'dbn') for gradients
    
    ## check if dictionary_structure is empty
    if L == 0:
        print('*** ERROR : empty dictionary ***')
        
    else :
        start = 0
        end = 0
        ## go through dictionary key by key, converting vector values for key into original dimensions of values (n,m)
        for key in dictionary_keys :
            end = end + np.prod(dictionary_structure[key])                            ## get lenght of vector (n*m) to convert into array (n,m) under key into new_dict
            new_dict[key] = np.reshape(vector[start:end], dictionary_structure[key])  ## reshape segment from vector of lenght (end - start) and reshape into original array (n,m)
            start = end                                                               ## next segment starts and current end
   
    return new_dict
'''

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    # Divide x by its norm.
    x = x / x_norm

    return x


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
    
    return v


#####################################
#####   ACTIVATION FUNCTIONS    #####
#####################################

def Identity(Z):
    A = Z
    return A

def dIdentity(dA, Z):
    A = 1
    return A

def Sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def dSigmoid(dA, Z):
    s = Sigmoid(Z)
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def Relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def dRelu(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z<0] = 0                   # When z < 0, set dz to 0 as well. 
    assert (dZ.shape == Z.shape)
    return dZ

def Leaky_relu(Z):
    A = np.maximum(0.01*Z, Z)
    return A

def dLeaky_relu(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z<0] = 0.01                # When z < 0, set dz to 0.01. 
    assert (dZ.shape == Z.shape)
    return dZ

def TanH(Z):
    A = np.tanh(Z)
    return A

def dTanH(dA, Z):
    dZ = dA * (1 - np.tanh(Z)**2)
    return dZ
        
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims = True)
    A = x_exp / x_sum
    return A



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


#####################################
######     MAKE MINIBATCHES    ######
#####################################


def make_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
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
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (num_complete_minibatches * mini_batch_size) : m]
        mini_batch_Y = shuffled_Y[:, (num_complete_minibatches * mini_batch_size) : m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
   
    return mini_batches

#####################################
######   ACTIVATION FUNCTIONS  ######
#####################################
    

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

    
######################################
##### LINEAR FORWARD PROPAGATION #####
######################################
    

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


##########################################
#### BACKWARD PROPAGATION FUNCTIONS  #####
##########################################


def activate_backward(dA, activation_cache):
    """
    Generic activation function
    Launches proper activation function according to 'activation' value from Layer_cache
    Allows the use of a specific activation function at each layer
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
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
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
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
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
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
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
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
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
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    Yhat -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    caches -- caches output from forward_propagation_with_dropout()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
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

'''
def gradient_check(parameters, gradients, X, Y, epsilon = 1e-8):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    W, b = parameters
    dA, dW, db = gradients
    grads_vector = []
    num_parameters = len(W)
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        thetaplus_W = np.copy(W)                                 # Step 1
        thetaplus_W += epsilon                                   # Step 2
        thetaplus_b = np.copy(b)                                 # Step 1
        thetaplus_b += epsilon                                   # Step 2
        thetaplus = (thetaplus_W, thetaplus_b)
        YhatL, _ = L_model_forward_propagation(X, thetaplus)
        J_plus[i] = compute_cross_entropy_cost(YhatL, Y)
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus_W = np.copy(W)                                 # Step 1
        thetaminus_W += epsilon                                   # Step 2
        thetaminus_b = np.copy(b)                                 # Step 1
        thetaminus_b += epsilon                                   # Step 2
        thetaminus = (thetaminus_W, thetaminus_b)
        YhatL, _ = L_model_forward_propagation(X, thetaminus)
        J_minus[i] = compute_cross_entropy_cost(YhatL, Y)

        grads_vector.append(dA[i])
        grads_vector.append(dW[i])
        grads_vector.append(db[i])
        
        print("Grads vector shape:", grads_vector.shape)
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grads_vector - gradapprox)                    # Step 1'
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator                              # Step 3'

    if difference > epsilon:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
'''

###############################################
### UPDATE PARAMETERS FROM GRADIENT DESCENT ###
###############################################

def update_parameters(m, parameters, grads, learning_rate, lambd):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
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
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
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
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
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


def Train_L_layers_DNN_model(X, Y, layers_dims, hyperparams):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    hyperparams -- list of hyperparameters in order num_iterations, print_cost, learning_rate, beta1, beta2, epsilon, lambd, keep_prob, initialization
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    ## initialize internal variables
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    np.random.seed(seed)
    t = 0                            # initializing the counter required for Adam update
    grads = []      ## to gather a dictionary of gradients from backprop
    costs = []      ## to keep track of the loss
    m = X.shape[1]  ## number of training samples
    
    ## Get hyperparameters
    ## DEFAULTS :
        ## num_iterations = 15000, print_cost = True
        ## learning_rate = 0.01, beta1 = .90, beta2 = 1, epsilon = 10**-8
        ## lambd = 0, keep_prob = 1, initialization = "he"
    n_epoch, print_cost, mini_batch_size, acthid, actout, optimizer, learning_rate, beta1, beta2, epsilon, lambd, keep_prob, initialization = hyperparams
    show_hyperparameters(hyperparams)
    
    # Initialize weights, bias and gradients parameters
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims, initialization)    
    
    # Initialize the optimizer
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


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    classes -- list of classes
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

"""
####################################################################################################
####################################################################################################
"""
