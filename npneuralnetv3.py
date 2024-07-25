'''
DATA WILL BE ORGANIZED LIKE THIS:
Activations will be matrices and have the COLUMNS correspond to individual training examples

Weights will be matrices and have the ROWS correspond to individual neurons (output activations);
Each weight has a weight for the number of input neurons

Biases will be an arrays that can be broadcasted to all training examples (expand horizontally);
they will be of size n * 1 (a column vector with each bias corresponding to one neuron) 

The first column of each set contains all of the correct labels
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


def get_data():
    #Reads files from directory and uses them to create numpy arrays
    train_file = r"archive (1)\mnist_train.csv"
    test_file = r"archive (1)\mnist_test.csv"
    training_set = pd.read_csv(train_file)
    training_set = np.array(training_set)
    test_set = pd.read_csv(test_file)
    test_set = np.array(test_set)

    #Creates a dev set by randomly selecting 15000 training examples
    np.random.shuffle(training_set)
    dev_set = np.array(training_set[:15000])
    training_set = training_set[15000:]

    #Transposes data to fit standard data normalization
    training_set = np.transpose(training_set)
    dev_set = np.transpose(dev_set)
    test_set = np.transpose(test_set)

    #Seperates Labels from input values
    training_set_y = training_set[0]
    training_set_x = training_set[1:]
    dev_set_y = dev_set[0]
    dev_set_x = dev_set[1:]
    test_set_y = test_set[0]
    test_set_x = test_set[1:]

    #Data Normalization - have to normalize dev/test values as well
    train_set_x_mean = np.mean(training_set_x)
    train_set_x_std = np.std(training_set_x)
    training_set_x = (training_set_x - train_set_x_mean)/train_set_x_std

    dev_set_x_mean = np.mean(dev_set_x)
    dev_set_x_std = np.std(dev_set_x)
    dev_set_x = (dev_set_x - dev_set_x_mean)/dev_set_x_std

    test_set_x_mean = np.mean(test_set_x)
    test_set_x_std = np.std(test_set_x)
    test_set_x = (test_set_x - test_set_x_mean)/test_set_x_std

    #Reshapes y values to a column vector
    training_set_y = training_set_y.reshape(45000,1)
    dev_set_y = dev_set_y.reshape(15000, 1)
    test_set_y = test_set_y.reshape(10000, 1)

    return training_set_x, training_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y




''' Variables for shape - n corresponds to features, m corresponds to examples
n, m = training_set_x.shape
'''

def init_params(n_x ,n_h1, n_h2, n_y):
    '''
    Takes in inputs for the number of neurons in the input, first hidden, second hidden, and output layers
    n_x = number of neurons in input layer
    n_h1 = number of neurons in hidden layer 1
    n_h2 = number of neurons in hidden layer 2
    n_y = number of neurons in output layer
    '''

    #First layer has 15 neurons which each have 784 inputs
    W1 = 0.25 * (np.random.rand(n_h1, n_x) - 0.5)
    b1 = 0.25 * (np.random.rand(n_h1, 1) - 0.5)

    #Second layer has 10 neurons which each have 15 neurons
    W2 = 0.25 * (np.random.rand(n_h2, n_h1) - 0.5)
    b2 = 0.25 * (np.random.rand(n_h2, 1) - 0.5 )

    #Output layer has 10 neurons which each have 15 neurons
    W3 = 0.25 * (np.random.rand(n_y, n_h2) - 0.5)
    b3 = 0.25 * (np.random.rand(n_y, 1) - 0.5)

    #Shove everything into a dictionary
    parameters = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3
    }

    return parameters

def relu(activations):
    '''
    Inputs: an array of activations of y x m where y is the number of neurons in the layer and m is the number
    of training examples

    Outputs: a nonlinear conversion of those values where anything below 0 is converted to zero; 
    an array of the same dimensions
    '''
    return np.maximum(0, activations)

def softmax(activations):
    '''
    Inputs: A matrix of activations where columns correspond to individual training examples
    Outputs: A softmax conversion of all the values within the matrix
    '''
    
    max_vals = np.max(activations, axis = 0) #Stability line to combat NaN results
    softmax_mat = np.exp(activations - max_vals) #Exponentiate all values
    sum_exps = np.sum(softmax_mat, axis = 0) #Creates an array where each value is the sum of a column
    softmax_output = softmax_mat/sum_exps

    return softmax_output

def one_hot_enc(set):
    '''
    Inputs: A column vector (y) of labels 0-9
    Outputs: A one-hot encoded version of the same thing; a matrix of size (m, 10)
    '''
    m = set.shape[0]

    encoded = np.eye(10)[set]
    encoded = encoded.reshape(m,10)

    return encoded.T

def forward_pass(parameters, X):
    
    #Gets Parameters from input parameter dictionary
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    #Compute the sequence of forward prop
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)

    #Shoves everything into a dictionary to be accessed later
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }

    return cache

def compute_cost(predicted, actual, epsilon):
    '''
    Inputs: A predicted output, the actual output, and epsilon - a value to substitue for zeros
    Outputs: The cost of a forward pass attempt with all training examples
    '''
    m = actual.shape[0]

    actual = one_hot_enc(actual) #Inputs the y labels (shape(45000,1)) and one hot encodes them to (shape(10, 45000))
    predicted = np.clip(predicted, epsilon, 1 - epsilon) #To prevent log(0)

    loss_matrix = actual * -np.log(predicted) #Multplies the one hot encoding to the predicted values
    cost = np.sum(loss_matrix)/m #Sum up all losses and divide by the number of training examples

    return cost


def dreLU(matrix):
    '''
    Inputs: A matrix of values (computed in backprop) to pass into this function
    Outputs: A matrix of booleans where values 0 or greater are considered true - which translates to 1
    '''
    return matrix >= 0

def backprop(parameters, cache, X, Y):

    m = Y.shape[0]
    
    #Get all parameters necessary
    A3 = cache["A3"] #Outputs
    A2 = cache["A2"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    Z1 = cache["Z1"]

    W3 = parameters["W3"]
    W2 = parameters["W2"]

    enc_y = one_hot_enc(Y)

    #Backprop Calculations
    dZ3 = A3 - enc_y
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis = 1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * dreLU(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * dreLU(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims=True)

    
    gradients = {
        "dW3": dW3,
        "db3": db3,
        "dW2": dW2,
        "db2": db2,
        "dW1": dW1,
        "db1": db1
    }

    return gradients

def update_params(gradients, parameters, learning_rate):

    #Access all necessary variables - creating deep copies for weights for safety
    '''
    W1 = copy.deepcopy(parameters["W1"])
    W2 = copy.deepcopy(parameters["W2"])
    W3 = copy.deepcopy(parameters["W3"])
    '''

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    dW1 = gradients["dW1"]
    dW2 = gradients["dW2"]
    dW3 = gradients["dW3"]
    db1 = gradients["db1"]
    db2 = gradients["db2"]
    db3 = gradients["db3"]

    W1 = W1 - (learning_rate) * (dW1)
    W2 = W2 - (learning_rate) * (dW2)
    W3 = W3 - (learning_rate) * (dW3)

    b1 = b1 - (learning_rate) * (db1)
    b2 = b2 - (learning_rate) * (db2)
    b3 = b3 - (learning_rate) * (db3)

    #Shove everything into a dictionary
    updated_params = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3
    }

    return updated_params


def model_predict(probs):
    '''
    Inputs: A model
    Outputs: A column vector of all of the max values of each column (each training example's prediction)
    '''

    m = probs.shape[1]
    
    predictions = np.argmax(probs, axis=0)
    predictions = predictions.reshape(m, 1)

    return predictions

def compute_accuracy(input_x, input_y):
    '''
    Inputs: Two Column Vector Matrices; predicted and actual values
    Returns: A single value determining the total percentages that match up
    '''
    len = input_x.shape[0]

    compare_matrix = input_x.T == input_y.T
    total_correct = compare_matrix.sum()

    return total_correct/len


def gradient_descent(iterations, learning_rate, X, Y, n_h1, n_h2):
    
    n_x = X.shape[0]
    n_y = Y.max() + 1 #Should include 10 classifications 0-9

    parameters = init_params(n_x ,n_h1, n_h2, n_y)
    print("Parameters Initialized")

    cost_history = []

    print("Starting Gradient Descent")
    for i in range(iterations):
        if i % 50 == 0 and i>1:
            print("Iterations", i)
            print("Cost", cost)

            predictions = model_predict(cache["A3"])
            current_acc = compute_accuracy(predictions, Y)
            print(f"Accuracy: {current_acc}%")

            print("================")

        cache = forward_pass(parameters, X)

        cost = compute_cost(cache["A3"], Y, 1e-3)
        cost_history.append(cost)

        gradients = backprop(parameters, cache, X, Y)

        parameters = update_params(gradients, parameters, learning_rate)

    final_params = parameters
    print("Final Parameters Stored")


    cost_history = np.array(cost_history, dtype=float)
    cost_history = cost_history.reshape((1,iterations))

    plt.plot([np.arange(iterations)],cost_history, "bo")
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()

    return cost_history, final_params


training_set_x, training_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y = get_data()

cost_history, final_params = gradient_descent(2000, 0.025, training_set_x, training_set_y, 50, 25)

dev_set_forward_pass = forward_pass(final_params, dev_set_x)
dev_set_predictions = model_predict(dev_set_forward_pass["A3"])
dev_set_accuracy = compute_accuracy(dev_set_predictions, dev_set_y)

print(f"Dev Set Accuracy: {dev_set_accuracy}%")