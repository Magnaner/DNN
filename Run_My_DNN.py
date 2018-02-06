# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:48:46 2017

@author: Éric Magnan (ericmagnan.ca)
Code inspired from Andrew Ng Stanford University
Deeplearning.ai Specialization courses on Coursera
"""

#####################################
## LOADING REQUIRED PYTHON MODULES ##
#####################################

from My_datasets_loaders import *    ## to load & plot training and testing datasets
import matplotlib.pyplot as plt      ## for plotting images inline
from My_DNN_Model import *           ## DNN model main code

%matplotlib inline

#############################
##    SELECT A DATASET     ##
#############################

""" TO LOAD ONE OF SKLEARN'S PLANAR DATASETS  """
""" circles, moons, blobs or biclusters """
## change the label between quotes in load_dataset() to choose the points model of your choice from sklearn
train_x, train_y, test_x, test_y, classes, superclasses = load_dataset('moons')

""" TO LOAD CATS IMAGES DATASET """
""" select and run lines between doublequotes below """
"""
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset('cats')

# RESHAPE DATA
# Reshape the training and test examples 
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

## STANDARDIZE DATA
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
"""

########################################
## DEFINE YOUR DNN LAYERS DIMENSIONS  ##
########################################
## this example is for a 4 hidden layers + 1 output layer deep neural net
## first layer is for input "X"
layers_dims = [train_x.shape[0], 20, 10, 10, 5, 1] #  N-layers model  


########################################
##      SET YOUR HYPERPARAMETERS      ##
########################################

""" Adjust default hyperparameters below or execute default settings """

### FOR HYPERPARAMETERS BY DEFAULT, RUN THE FOLLOWING 3 LINES
hyperparams = set_default_hyperparameters()
acthid = hyperparams[3]
actout = hyperparams[4]

### TO USE SELF DEFINED HYPERPARAMETERS, ADJUST AND EXECUTE THE BLOCK BELOW

"""
Usually proposed values :
learning_rate = 0.0007, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, n_epochs = 10000, print_cost = True
"""
## BASIC HYPERPARAMETERS
n_epoch = 3000        ## higher number = longer training time but better accuracy
print_cost = True     ## set to True if you want to see the training error graph
mini_batch_size = 300 ## could use 64, 128, 256
init_type = "he"      ## "random", "he", "xavier" or "bengio"
acthid = "relu"       ## hidden layers activation function: relu, tanh, leaky_relu or sigmoid 
actout = "sigmoid"    ## output layer activation function: sigmoid, softmax or tanh
Lr = 0.001            ## Training learning rate

## OPTIMNIZERS
optimizer = "adam"    ## gradient optimizer: "gd", "adam", "momentum"
beta1 = 0.9           ## 1 = off, for exponential decay for momentum or first moment estimates in Adam
beta2 = 0.999         ## 1 = off, for exponential decay for the second moment estimates in Adam
epsilon = 1e-8        ## keep as-is to prevent division by zero in Adam updates

## REGULARIZATION METHODS
lambd = 0.5           ## 0 = off, regularization hyperparameter
keep_prob = 1.        ## 1 = off, non-zero value < 1.0, represents % neurons kept activated for dropout

## DEFINE HYPERPARAMETERS LIST FOR DNN MAIN CODE    
hyperparams = (n_epoch, print_cost, mini_batch_size, acthid, actout, optimizer, Lr, beta1, beta2, epsilon, lambd, keep_prob, init_type)


########################################
##       TRAIN THE DNN MODEL          ##
########################################

parameters = Train_L_layers_DNN_model(train_x, train_y, layers_dims, hyperparams)


### DISPLAY TRAINING RESULTS 
# ACCURACY
p_train = predict(train_x, parameters, acthid, actout, Bool=True)
p_train_scores = predict(train_x, parameters, acthid, actout, Bool=False)
score = np.sum(p_train == train_y) / train_x.shape[1]
print("Accuracy: "  + str(score))
m1 = train_x.shape[1]
bad1 = []
for i in range(m1):
    if p_train[0,i] != train_y[0,i]:
        bad1.append(i)
print(len(bad1),"wrong predictions out of",m1)

# TO SEE SPECIFIC MISCLASSIFICATIONS
''' FOR PLANAR DATASETS '''
for i in range(0,p_train_scores.shape[1]):
    if p_train[0,i] != train_y[0,i]:
        print("#",i,":",round(p_train_scores[0,i]*100,2),"% as", bool(p_train[0,i]),"when",bool(train_y[0,i]))

plot_decision_boundary(lambda x: predict(x.T, parameters, acthid, actout), train_x, train_y)
      
''' FOR IMAGE DATASETS ONLY '''
for i in range(len(bad1)):
    k = bad1[i]
    plt.imshow(train_set_x_orig[k])
    plt.show()
    print("Prédicted is {} with {}% confidence while thruth is {}".format(bool(p_train[0,k]), round(p_train_scores[0,k]*100,2), bool(train_y[0,k])))

## TO PLOT PREDICTION BOUNDARIES


########################################
##        TEST THE DNN MODEL          ##
########################################

p_test = predict(test_x, parameters, acthid, actout, Bool=True)
p_test_scores = predict(test_x, parameters, acthid, actout, Bool=False)


### DISPLAY TEST RESULTS 
# ACCURACY
score = np.sum(p_test == test_y) / test_x.shape[1]
print("Accuracy: "  + str(score))
m2 = test_x.shape[1]
bad2 = []
for i in range(m2):
    if p_test[0,i] != test_y[0,i]:
        bad2.append(i)
print(len(bad2),"wrong predictions out of",m2)

# TO SEE SPECIFIC MISCLASSIFICATIONS
''' FOR PLANAR DATASETS '''
for i in range(0,p_test_scores.shape[1]):
    if p_test[0,i] != test_y[0,i]:
        print("#",i,":",round(p_test_scores[0,i]*100,2),"% as", bool(p_test[0,i]),"when",bool(test_y[0,i]))

plot_decision_boundary(lambda x: predict(x.T, parameters, acthid, actout, Bool=True), test_x, test_y)
        
''' FOR IMAGE DATASETS ONLY '''
for i in range(len(bad2)):
    k = bad2[i]
    plt.imshow(test_set_x_orig[k])
    plt.show()
    print("Prédicted is {} with {}% confidence while thruth is {}".format(bool(p_test[0,k]), round(p_test_scores[0,k]*100,2), bool(test_y[0,k])))

###############################
## TEST WITH YOUR OWN IMAGE  ##
###############################
    
my_image = "my_image.jpg"   # change this to the name of your image file 

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255
my_predicted_image = predict(my_image, parameters, acthid, actout, Bool=True)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
