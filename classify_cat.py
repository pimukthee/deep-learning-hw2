# -*- coding: utf-8 -*-
"""
    Cat VS non-cat classification
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py  # for reading data in h5 format
import feedforward_nn as fnn

########################
#   Dataset functions  #
########################
def load_cat_data(verbosity=False):
    """
    You are given the "Cat vs non-Cat" dataset (in the dataset folder) containing:
        - a training set of m_train images labeled as cat (1) or non-cat (0)
        - a test set of m_test images labeled as cat and non-cat
        - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).    
    """    
    
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
        
    # Print dataset's detail
    if (verbosity):
        m_train = train_x.shape[0]
        num_px  = train_x.shape[1]
        m_test  = test_x.shape[0]
        
        print("Dataset:")
        print("  Number of training examples: " + str(m_train))
        print("  Number of testing examples: " + str(m_test))
        print("  Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("  train_x_shape: " + str(train_x.shape))
        print("  train_y shape: " + str(train_y.shape))
        print("  test_x shape: " + str(test_x.shape))
        print("  test_y shape: " + str(test_y.shape))    
    
    return train_x, train_y, test_x, test_y, classes

def preprocess_cat_data(train_x, test_x, verbosity=False):
    """
    Reshape and normalize the images before inputting them into the network
    Argument:
    train_x -- original training data
    test_x  -- original testing data    
    """
    
    if (verbosity):
        print("Reshape dataset from ...") 
        print("  train_x.shape = " + str(train_x.shape))
        print("  test_x.shape  = " + str(test_x.shape))

    # Reshape the training and test examples 
    train_x = train_x.reshape(train_x.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x = test_x.reshape(test_x.shape[0], -1).T
    
    # Normalize data to have feature values between 0 and 1.
    train_x = train_x/255.0
    test_x = test_x/255.0

    if (verbosity):
        print("To ...")
        print("  train_x.shape = " + str(train_x.shape))
        print("  test_x.shape  = " + str(test_x.shape))
    
    return train_x, test_x

def preview_cat_data(X, Y, index):
    """
    Show the original data's image 
    Argument:
       X -- original data
       Y -- label of the original
       index -- the index of the image to be previewed
    """
    
    assert len(X.shape) == 4, "Invalid data's dimensions."

    print("Preview the image of index = " + str(index))
    print("  y = " + str(Y[0,index]))    
    plt.imshow(X[index])    

################################
#        Main function         #
################################    
if __name__ == '__main__':
    
    np.random.seed(1)
    
    # Load data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_cat_data(True)
    #preview_cat_data(test_x_orig, test_y, 11)  
    
    # Preprocess data
    train_x, test_x = preprocess_cat_data(train_x_orig, test_x_orig)
    n = train_x.shape[0]   # Number of input features
    
    
    #######################
    ### START CODE HERE ### 
    #######################         
    # When running one model, just comment out the others
    
    # Run 1-layer model (logistic regression)
    # parameters = fnn.L_layer_model(train_x, train_y, [n, 1], learning_rate = 0.005, num_iterations = 2400)
    
    # Run 2-layer model 
    # parameters = fnn.L_layer_model(train_x, train_y, [n, 7, 1], learning_rate = 0.0075, num_iterations = 2400)

    # Run 4-layer model 
    parameters = fnn.L_layer_model(train_x, train_y, [n, 20, 7, 5, 1], learning_rate = 0.0075, num_iterations = 2400, print_cost=True)
    
    #####################
    ### END CODE HERE ###
    #####################

    # Report
    train_yhat = fnn.predict(train_x, parameters)
    test_yhat  = fnn.predict(test_x, parameters)    
    accuracy_train = fnn.compute_accuracy(train_yhat, train_y)
    accuracy_test  = fnn.compute_accuracy(test_yhat, test_y)
    print("Accuracy on the training set = " + str(accuracy_train*100) + "%")
    print("Accuracy on the test set = " + str(accuracy_test*100) + "%")