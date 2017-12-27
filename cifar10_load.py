# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:30:32 2017

@author: Éric

Load CIFAR-10 dataset in python3
CIFAR-10
Train: 50,000
Test: 10,000
#Classes: 10
Download: CIFAR-10 Python version


data -- 10000x3072 numpy array of uint8S. Each row of the array stores a 32x32 color image.
           [1024: Red][1024: Green][1024: Blue] ==> 3027

labels -- 10000, the range 0 - 9 [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data


def load_cifar10():
    '''Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3'''
    train_filenames = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    test_filenames = []
    data_dir = 'Datasets/CFAR-10/cifar-10-batches-py'

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
            train_labels = data_dic['labels']
            train_filenames = data_dic['filenames']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
            train_labels += data_dic['labels']
            train_filenames += data_dic['filenames']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    test_filenames += test_data_dic['filenames']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)
    train_filenames = np.array(train_filenames)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)
    test_filenames = np.array(test_filenames)
    
    classes = load_class_names_cifar10(data_dir)
    superclasses = []
    
    ## INFO ABOUT USEFULL FUNCTIONS OF THIS MODULE
    print()
    print('Availiable usefull functions are :')
    print('showclasses_cifar10() prints images classes by index number')
    print('showme_cifar10(ndx) gets info on and plots training image number ndx')
    print('findall_classes_cifar10(ndx) gets how many training images are of class ndx, returns corresponding images index')
    print()
    print('load_cifar10() returns: train_data, train_labels, test_data, test_labels, classes, superclasses')

    return train_data, train_labels, test_data, test_labels, classes, superclasses


def load_class_names_cifar10(data_dir):
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """
    # Load the class-names from the pickled file.
    raw = []
    names = []
    raw = unpickle(data_dir + '/batches.meta')
    names = raw['label_names']
    
    # Convert from binary strings.
    ## names = [x.decode('utf-8') for x in raw]

    return names


def showme_cifar10(ndx):
    print('Training image ' + str(ndx) + ' is of label ' + str(train_class_labels[ndx]) + ' and class ' + classes[train_class_labels[ndx]])
    print('Image is :')
    plt.imshow(train_data[ndx])
    plt.show()
  
    
def showclasses_cifar10():
    for i in range(0,len(classes)):
        print('Label ' + str(i) + ' is class ' + classes[i])


def showsuperclasses_cifar10():
    print('There are no superclasses for CIFAR-10.')

        
def findall_classes_cifar10(ndx):
    selection = []
    for i in range(0,len(train_class_labels)):
        if train_class_labels[i] == ndx:
            selection.append(i)
    print(str(len(selection)) + ' images of ' + classes[ndx] + ' found.')

    """
    for i in range(0,len(selection)):
        plt.imshow(train_data[selection[i]])
        
    plt.show()
    """    
    return selection