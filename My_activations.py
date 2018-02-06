# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:52:25 2017

@author: Eric Magnan
"""
###########################
#####    L O A D      #####
###########################

import numpy as np

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
