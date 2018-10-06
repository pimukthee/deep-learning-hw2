# -*- coding: utf-8 -*-
"""
Deep L-layer feedforward neural network

"""

import numpy as np
import matplotlib.pyplot as plt                     
from testcases import *

####################################
#     Parameter initialization     #
####################################
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network           
                  e.g. layer_dims = [15, 6, 3, 1] defines the number of nodes in 4-layer feedforward neural network
                  with 15 nodes in the input layer, 6 nodes in the first hidden layer,
                  3 nodes in the second hidden layer, and 1 node of output layer.
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):                
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))        
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))  
    return parameters

###############################
#     Forward propagation     #
###############################
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Z = W*A + b

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    #######################
    ### START CODE HERE ### (≈ 1 line of code)    
    #######################    
    Z = np.matmul(W, A) + b
    #######################
    ###  END CODE HERE  ###
    #######################
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def sigmoid_forward(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    Z -- Z will be useful during backpropagation
    """
    #######################
    ### START CODE HERE ### (≈ 1 line of code)    
    #######################    
    A = 1 / (1 + np.exp(-Z))
    #######################
    ###  END CODE HERE  ###
    #######################    
    return A, Z

def relu_forward(Z):
    """
    Implements the ReLU activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of max{0,z}, same shape as Z
    Z -- Z will be useful during backpropagation
    """    
    #######################
    ### START CODE HERE ### (≈ 1 line of code)    
    #######################    
    A = abs(Z) * (Z > 0)
    #######################
    ###  END CODE HERE  ###
    #######################     
    
    return A, Z

def tanh_forward(Z):
    """
    Implements the tanh activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of tanh(z), same shape as Z
    Z -- Z will be useful during backpropagation
    """   
    #######################
    ### START CODE HERE ### (≈ 1 line of code)    
    #######################    
    A = 2*sigmoid_forward(2*Z) - 1
    #######################
    ###  END CODE HERE  ###
    #######################       
    return A, Z

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Z = W*A_prev + b --> A = activation_function(Z)

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "tanh"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################e
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid_forward(Z)
        #####################
        ### END CODE HERE ###
        #####################        
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################     
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_forward(Z)
        #####################
        ### END CODE HERE ###
        #####################

    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################       
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh_forward(Z)
        #####################
        ### END CODE HERE ###
        #####################
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU or TAHN]*(L-1). And add "cache" to the "caches" list.
    for l in range(1, L):
    
        A_prev = A 
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################     
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
        #####################
        ### END CODE HERE ###
        #####################
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    #######################
    ### START CODE HERE ### (≈ 2 lines of code)
    #######################
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    #####################
    ### END CODE HERE ###
    #####################
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

###############################
#       Evaluate COST        #
###############################
def compute_cost(AL, Y):
    """
    Implement the cost function defined by the cost of logistic regression.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    #######################
    ### START CODE HERE ### (≈ 1 lines of code)
    #######################
    cost = -(1.0/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))   # Use the cost of logistic regression
    #####################
    ### END CODE HERE ###
    #####################
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

###############################
#    Backward propagation     #
###############################
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    #######################
    ### START CODE HERE ### (≈ 3 lines of code)
    #######################
    dW = (1/m) * np.matmul(dZ, A_prev.T)
    db = (1/m) * np.matmul(dZ, np.ones((m, 1)))
    dA_prev = np.matmul(W.T, dZ)
    #####################
    ### END CODE HERE ###
    #####################
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def sigmoid_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape (= dJ/dA)
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z (= dJ/dZ)
    """    

    Z = activation_cache

    #######################
    ### START CODE HERE ### (≈ 1 line of code)
    #######################        
    A = 1 / (1 + np.exp(-Z))
    dZ = dA * (A * (1-A))
    #####################
    ### END CODE HERE ###
    ##################### 
    
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single RELU unit.
    (ReLU is defined as A = max{0, Z}. Your job is to compute dJ/dZ)

    Arguments:
    dA -- post-activation gradient, of any shape (= dJ/dA)
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z (= dJ/dZ)
    """
    
    Z = activation_cache        
    #######################
    ### START CODE HERE ### (≈ 2 lines of code)
    #######################
    # Hint: When z <= 0, you should set dz to 0 as well.    
    dZ = dA *  (1*(Z>0))
    #####################
    ### END CODE HERE ###
    #####################   
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanh_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single TANH unit.
    (ReLU is defined as A = tanh(Z). Your job is to compute dZ)

    Arguments:
    dA -- post-activation gradient, of any shape (= dJ/dA)
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z (= dJ/dZ)
    """   
    
    Z = activation_cache 
    
    #######################
    ### START CODE HERE ### (≈ 1-2 lines of code)
    #######################        
    A = np.tanh(Z)
    dZ = dA * (1 - A*A)
    #####################
    ### END CODE HERE ###
    #####################     
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "tanh"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #####################
        ### END CODE HERE ###
        #####################
        
    elif activation == "sigmoid":
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #####################
        ### END CODE HERE ###
        #####################

    elif activation == "tanh":
        #######################
        ### START CODE HERE ### (≈ 2 lines of code)
        #######################
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #####################
        ### END CODE HERE ###
        #####################
    
    return dA_prev, dW, db
       
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # 1) Initializing the backpropagation
    #######################
    ### START CODE HERE ### (1 line of code)
    #######################    
    dAL = -(Y/AL) + ((1-Y)/(1-AL))
    #####################
    ### END CODE HERE ###
    #####################
    
    # 2) Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    #######################
    ### START CODE HERE ### (approx. 2 lines)
    #######################
    current_cache = caches[L-1]    
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    #####################
    ### END CODE HERE ###
    #####################
    
    # 3) Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU/TAHN -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        #######################
        ### START CODE HERE ### (approx. 5 lines)
        #######################
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp =  linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp       
        #####################
        ### END CODE HERE ###
        #####################

    return grads

########################################
#   One-step gradient descent update   #
########################################
def update_parameters(parameters, grads, learning_rate):
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
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    #######################
    ### START CODE HERE ### (≈ 3 lines of code)
    #######################
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads['db' + str(l+1)]
    #####################        
    ### END CODE HERE ###
    #####################
    return parameters

    
###############################
#    Overall L-layer model    #
###############################
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    print("Training " + str(len(layers_dims)-1) +"-layer network: layers_dims = " + str(layers_dims) )

    np.random.seed(3)
    costs = []                         # keep track of cost    
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        #######################
        ### START CODE HERE ### (≈ 1 line of code)
        #######################        
        AL, caches = None
        #####################
        ### END CODE HERE ###
        #####################
        
        # Compute cost.
        #######################
        ### START CODE HERE ### (≈ 1 line of code)
        #######################        
        cost = None
        #####################
        ### END CODE HERE ###
        #####################
    
        # Backward propagation.
        #######################
        ### START CODE HERE ### (≈ 1 line of code)
        #######################        
        grads = None
        #####################
        ### END CODE HERE ###
        #####################
 
        # Update parameters.
        #######################
        ### START CODE HERE ### (≈ 1 line of code)
        #######################        
        parameters = None
        #####################
        ### END CODE HERE ###
        #####################
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("   Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, parameters):
    """
    This function is used to predict the results of a L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    Y_hat -- predictions for the given dataset X (either 0 or 1)
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    Y_hat = np.zeros((1,m))
    
    # Forward propagation
    AL, caches = L_model_forward(X, parameters)
    
    # convert prediction (values between 0 and 1) to 0/1 predictions    
    #######################
    ### START CODE HERE ### (≈ 5 line of code)
    #######################    
    # Convert AL to 0 or 1 by thresholding at 0.5
    Y_hat = None  
    #####################
    ### END CODE HERE ###
    #####################
        
    return Y_hat

def compute_accuracy(Y_hat, Y):
    """
    Compute accuracy of the prediction result. 
    
    Arguments:
     Y -- ground truth / target {0,1}
     Y_hat -- predictions {0,1}
     
    Return:
     accuracy -- Number of correct predictions / Total number of predictions
    """
    
    assert ( Y_hat.shape == Y.shape)
    #######################
    ### START CODE HERE ### (≈ 1 line of code)
    #######################     
    accuracy = None
    #####################
    ### END CODE HERE ###
    #####################
    
    return accuracy
    

################################
#        Main function         #
################################    
if __name__ == '__main__':
    
    np.random.seed(1)                

    # BELOW ARE SOME USEFUL TEST CASES TO VERIFY THE OUTPUTS OF YOUR FUNCTIONS

    # Test case for linear_forward()
    print('-'*50)  
    print('Z, linear_cache = linear_forward()')
    A, W, b = linear_forward_test_case()
    Z, linear_cache = linear_forward(A, W, b)
    print("Z = " + str(Z))
    
    # Test case for linear_activation_forward()
    print('-'*50)  
    print('linear_activation_forward()')
    A_prev, W, b = linear_activation_forward_test_case()
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    
    # Test case for L_model_forward()
    print('-'*50) 
    print('L_model_forward()')
    X, parameters = L_model_forward_test_case_2hidden()
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))

    # Test case for compute_cost()
    print('-'*50) 
    print('compute_cost()')
    Y, AL = compute_cost_test_case()
    print("cost = " + str(compute_cost(AL, Y)))

    # Test case for linear_backward()
    print('-'*50) 
    print('linear_backward()')
    dZ, linear_cache = linear_backward_test_case()
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))

    # Test case for linear_activation_backward()
    print('-'*50)  
    print('linear_activation_backward()')
    dAL, linear_activation_cache = linear_activation_backward_test_case()
    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")
    dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    # Test case for L_model_backward()
    print('-'*50)  
    print('L_model_backward()')    
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    print_grads(grads)

    # Test case for update_parameters_test_case()
    print('-'*50)  
    print('update_parameters()')
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))


    
