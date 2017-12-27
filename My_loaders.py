# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:17:17 2017

@author: Éric
"""

########################
### IMPORT LIBRARIES ###
########################

import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import h5py
import cifar10_load as cf10
import cifar100_load as cf100


#############################
### LOAD SKLEARN DATASETS ###
#############################

def load_dataset(name):
    print('Dataset is for : ', name)
    if name == 'moons':
        return load_moons()
    
    elif name == 'circles':
        return load_circles()
    
    elif name == 'blobs':
        return load_blobs()
    
    elif name == 'biclusters':
        return load_biclusters()
    
    elif name == 'cats':
        return load_cats_images()
    
    elif name == 'cifar10':
        return cf10.load_cifar10()
    
    elif name == 'cifar100':
        return cf100.load_cifar100()
    
    else:
        print('Unknown data set specified. Load incomplete !')
        

def load_moons():
    np.random.seed(3)
    classes = []
    superclasses = ['none']
    
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=600, noise=.2) #300 #0.2 
    test_X = np.zeros(train_X.shape)
    test_Y = np.zeros(train_Y.shape)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    test_X, test_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
   
    
    print('BEWARE !!  This dataset has no test set, no classes and no superclasses.')
    print('test_X, test_Y, classes, superclasses are set to [].')
    
    return train_X, train_Y, test_X, test_Y, classes, superclasses


def load_circles():
    np.random.seed(3)
    classes = []
    superclasses = ['none']
    
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=600, noise=.2) #300 #0.2 
    test_X = np.zeros(train_X.shape)
    test_Y = np.zeros(train_Y.shape)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    test_X, test_Y = sklearn.datasets.make_circles(n_samples=300, noise=.2) #300 #0.2 
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
   
    print('BEWARE !!  This dataset has no test set, no classes and no superclasses.')
    print('test_X, test_Y, classes, superclasses are set to [].')
    
    return train_X, train_Y, test_X, test_Y, classes, superclasses


def load_blobs():
    np.random.seed(3)
    classes = []
    superclasses = ['none']
    
    train_X, train_Y = sklearn.datasets.make_blobs(n_samples=600)
    test_X = np.zeros(train_X.shape)
    test_Y = np.zeros(train_Y.shape)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    test_X, test_Y = sklearn.datasets.make_blobs(n_samples=300) #300 #0.2 
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
   
    print('BEWARE !!  This dataset has no test set, no classes and no superclasses.')
    print('test_X, test_Y, classes, superclasses are set to [].')
    
    return train_X, train_Y, test_X, test_Y, classes, superclasses


def load_biclusters():
    np.random.seed(3)
    classes = []
    superclasses = ['none']
    
    train_X, train_Y = sklearn.datasets.make_biclusters(n_clusters=600)
    test_X = np.zeros(train_X.shape)
    test_Y = np.zeros(train_Y.shape)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    test_X, test_Y = sklearn.datasets.make_biclusters(n_clusters=300) 
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
   
    print('BEWARE !!  This dataset has no test set, no classes and no superclasses.')
    print('test_X, test_Y, classes, superclasses are set to [].')
    
    return train_X, train_Y, test_X, test_Y, classes, superclasses


#############################
### LOAD LOCAL CAT IMAGES ###
#############################
    
def load_cats_images():
    train_dataset = h5py.File('Datasets/Cats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/Cats/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    ## Display dataset info
    m_train = train_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    m_test = test_set_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x_orig shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y_orig.shape))
    print ("test_set_x_orig shape: " + str(test_set_x_orig.shape))
    print ("test_y shape: " + str(test_set_y_orig.shape))
       
    # Display an example of a picture
    index = random.randint(1,m_train)
    plt.imshow(train_set_x_orig[index])
    print("It's a " + classes[train_set_y_orig[index]].decode("utf-8") +  " picture.")

    return  train_set_x_orig,  train_set_y_orig,  test_set_x_orig,  test_set_y_orig, classes
