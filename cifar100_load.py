# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:30:32 2017

@author: Éric

Load CIFAR-100 dataset in python3
CIFAR-100
Train: 500
Test: 100
#Classes: 100
#superclasses: 20
Download: CIFAR-100 Python version


data -- 10000x3072 numpy array of uint8S. Each row of the array stores a 32x32 color image.
           [1024: Red][1024: Green][1024: Blue] ==> 3027

classlabels -- 100, the range 0 - 9 [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
superclasslabels -- 20, the range 0 - 19
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data


def load_cifar100():
    '''Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3'''
    train_data = None
    test_data = None
    data_dir = 'Datasets\cifar-100-python\cifar-100-python'

    train_class_labels = []
    train_superclass_labels = []
    test_class_labels = []
    test_superclass_labels = []
 
    data_dic = unpickle(data_dir + "/train")
    train_data = data_dic['data']
    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
   
    train_class_labels = data_dic['fine_labels']
    train_class_labels = np.array(train_class_labels)
    
    train_superclass_labels = data_dic['coarse_labels']
    train_superclass_labels = np.array(train_superclass_labels)
 
    test_data_dic = unpickle(data_dir + "/test")
    test_data = test_data_dic['data']
    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    
    test_class_labels = test_data_dic['fine_labels']
    test_class_labels = np.array(test_class_labels)
    
    test_superclass_labels = test_data_dic['coarse_labels']
    test_superclass_labels = np.array(test_superclass_labels)
    
    class_names, superclass_names = load_class_names_cifar100()
    
    ## INFO ABOUT USEFULL FUNCTIONS OF THIS MODULE
    print()
    print('Availiable usefull functions are :')
    print('showclasses_cifar100() and showsuperclasses_cifar100(), print the classes by index number')
    print('showme_cifar100(ndx) gets info on and plots training image number ndx')
    print('findall_classes_cifar100(ndx) gets how many training images are of class ndx, returns corresponding images index')
    print('findall_superclasses_cifar100(ndx) gets how many training images are of superclass ndx, returns corresponding images index')
    print()
    print('load_cifar100() returns: train_data, train_class_labels, train_superclass_labels, test_data, test_class_labels, test_superclass_labels')

    return train_data, train_class_labels, train_superclass_labels, test_data, test_class_labels, test_superclass_labels

def load_class_names_cifar100():
    """
    Load the names for the classes in the CIFAR-100 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """
    # Load the class-names from the pickled file.
    raw = unpickle('Datasets\cifar-100-python\cifar-100-python/meta')
    class_names = raw['fine_label_names']
    superclass_names = raw['coarse_label_names']
    
    # Convert from binary strings.
    ## names = [x.decode('utf-8') for x in raw]

    return class_names, superclass_names


def showme_cifar100(ndx):
    print('Training image ' + str(ndx) + ' is of fine_label ' + str(train_class_labels[ndx]) + ' and of class name ' + class_names[train_class_labels[ndx]])
    print('Image is of superclass:' + superclass_names[train_superclass_labels[ndx]])
    plt.imshow(train_data[ndx])
    plt.show()
  
    
def showclasses_cifar100():
    for i in range(0,len(class_names)):
        print('Label ' + str(i) + ' for class ' + class_names[i])


def showsuperclasses_cifar100():
    for i in range(0,len(superclass_names)):
        print('Label ' + str(i) + ' is for superclass ' + superclass_names[i])
    

def findall_classes_cifar100(ndx):
    selection = []
    for i in range(0,len(train_class_labels)):
        if train_class_labels[i] == ndx:
            selection.append(i)
    print(str(len(selection)) + ' images of ' + class_names[ndx] + ' found.')

    """
    for i in range(0,len(selection)):
        plt.imshow(train_data[selection[i]])
        
    plt.show()
    """    
    return selection


def findall_superclasses_cifar100(ndx):
    selection = []
    for i in range(0,len(train_superclass_labels)):
        if train_superclass_labels[i] == ndx:
            selection.append(i)
    print(str(len(selection)) + ' images of ' + superclass_names[ndx] + ' found.')

    """
    for i in range(0,len(selection)):
        plt.imshow(train_data[selection[i]])
        
    plt.show()
    """    
    return selection
