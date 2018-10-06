# -*- coding: utf-8 -*-
"""
Binary classification on 2D artificial dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import feedforward_nn as fnn

def load_planar_dataset():
    """
    Data generator
    """
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary along with training examples (2D only)
    """
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    
    
################################
#        Main function         #
################################    
if __name__ == '__main__':
        
    train_x, train_y = load_planar_dataset()
    n = train_x.shape[0]
    # Training feedforward neural network (TANH[L-1]-->SIGMOID)
    #######################
    ### START CODE HERE ### 
    #######################       
    parameters = fnn.L_layer_model(train_x, train_y, [n, 4, 1], learning_rate = 1.2, num_iterations = 10000, print_cost=True)      
    #####################
    ### END CODE HERE ###
    #####################

    plot_decision_boundary(lambda x: fnn.predict(x.T, parameters), train_x, train_y)  
    
    train_yhat = fnn.predict(train_x, parameters)    
    accuracy_train = fnn.compute_accuracy(train_yhat, train_y)    
    print("Accuracy on the training set = " + str(accuracy_train*100) + "%")
