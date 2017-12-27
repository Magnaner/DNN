# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:48:46 2017

@author: Éric Magnan (ericmagnan.ca)
"""
from My_loaders import *
import matplotlib.pyplot as plt
from My_DNN_Final import *
from My_utils import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


%matplotlib inline

## LOAD DATA
""" circles, moons, blobs or biclusters """
## change the label between quotes in load_dataset() to choose the points model of your choice from sklearn
train_x, train_y, test_x, test_y, classes, superclasses = load_dataset('moons')

""" cats  """
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

## DEFINE MODEL LAYERS DIMENSIONS
layers_dims = [train_x.shape[0], 20, 10, 10, 5, 1] #  N-layers model  

## TRAIN MODEL
hyperparams = set_default_hyperparameters()
acthid = hyperparams[3]
actout = hyperparams[4]

parameters = Train_L_layers_DNN_model(train_x, train_y, layers_dims, hyperparams)

""" DISPLAY TRAINING RESULTS """
#print results
#print ("predictions: " + str(p))
#print ("true labels: " + str(y))
p_train = predict(train_x, parameters, acthid, actout, Bool=True)
p_train_Yhat = predict(train_x, parameters, acthid, actout, Bool=False)
score = np.sum(p_train == train_y) / train_x.shape[1]
print("Accuracy: "  + str(score))

## to print TRAIN images
m1 = train_x.shape[1]
bad1 = []
for i in range(m1):
    if p_train[0,i] != train_y[0,i]:
        bad1.append(i)
print(len(bad1),"wrong predictions out of",m1)

''' IF POINTS '''
for i in range(0,p_train_Yhat.shape[1]):
    if p_train[0,i] != train_y[0,i]:
        print("#",i,":",round(p_train_Yhat[0,i]*100,2),"% as", bool(p_train[0,i]),"when",bool(train_y[0,i]))
        
''' IF IMAGES '''
'''
for i in range(len(bad1)):
    k = bad1[i]
    plt.imshow(train_set_x_orig[k])
    plt.show()
    print("Prédicted is {} with {}% confidence while thruth is {}".format(bool(p_train[0,k]), round(p_train_Yhat[0,k]*100,2), bool(train_y[0,k])))
 '''
   
## To print sklearn datasets
plot_decision_boundary(lambda x: predict(x.T, parameters, acthid, actout), train_x, train_y)


""" DISPLAY TEST RESULTS """
#print results
#print ("predictions: " + str(p))
#print ("true labels: " + str(y))
p_test = predict(test_x, parameters, acthid, actout, Bool=True)
p_test_Yhat = predict(test_x, parameters, acthid, actout, Bool=False)
score = np.sum(p_test == test_y) / test_x.shape[1]
print("Accuracy: "  + str(score))

## to print TEST images
m2 = test_x.shape[1]
bad2 = []
for i in range(m2):
    if p_test[0,i] != test_y[0,i]:
        bad2.append(i)
print(len(bad2),"wrong predictions out of",m2)

''' IF POINTS '''
for i in range(0,p_test_Yhat.shape[1]):
    if p_test[0,i] != test_y[0,i]:
        print("#",i,":",round(p_test_Yhat[0,i]*100,2),"% as", bool(p_test[0,i]),"when",bool(test_y[0,i]))
        
''' IF IMAGES '''
'''
for i in range(len(bad2)):
    k = bad2[i]
    plt.imshow(test_set_x_orig[k])
    plt.show()
    print("Prédicted is {} with {}% confidence while thruth is {}".format(bool(p_test[0,k]), round(p_test_Yhat[0,k]*100,2), bool(test_y[0,k])))
'''

## To print sklearn datasets
plot_decision_boundary(lambda x: predict(x.T, parameters, acthid, actout, Bool=True), test_x, test_y)

"""
### RUN MODEL ###
## LOAD DATA
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# RESHAPE DATA
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

## STANDARDIZE DATA
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

## DEFINE MODEL DIMS
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model    

## RUN MODEL
hyperparams = set_hyperparameters()
parameters = Train_L_layers_DNN_model(train_X, train_Y, layers_dims, hyperparams)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

## TEST with my image
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""