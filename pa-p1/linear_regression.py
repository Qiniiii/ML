"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    _y = np.dot(X, np.transpose(w))
    err = np.sum(np.abs(_y - np.transpose(y))) / len(_y)
    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    X_T = np.transpose(X)
    aaa = np.dot(X_T, X)
    tem_XX = np.linalg.inv(aaa)
    w = np.dot(np.dot(tem_XX, X_T), np.transpose(y))
    return w


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################

    X_T = np.transpose(X)
    aaa = np.dot(X_T, X)
    I = np.identity(len(aaa))
    values = np.linalg.eigvals(aaa)
    while np.min(np.abs(values)) < 0.0001:
        aaa = aaa + 0.1 * I
        values = np.linalg.eigvals(aaa)
    tem_XX = np.linalg.inv(aaa)
    w = np.dot(np.dot(tem_XX, X_T), np.transpose(y))
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    #####################################################

    X_T = np.transpose(X)
    aaa = np.dot(X_T, X)
    I = np.identity(len(aaa))
    aaa = aaa + lambd * I
    tem_XX = np.linalg.inv(aaa)
    w = np.dot(np.dot(tem_XX, X_T), np.transpose(y))
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = 10 ** (-19)
    w = regularized_linear_regression(Xtrain, ytrain, 10 ** (-19))
    mea = mean_absolute_error(w, Xval, yval)
    for i in range(-18, 20):
        lambd = 10 ** i
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        if mean_absolute_error(w, Xval, yval) < mea:
            bestlambda = lambd
            mea = mean_absolute_error(w, Xval, yval)

    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    mapped_X=X
    for i in range(2,power+1):
        tem_X = np.float_power(X,i)
        mapped_X = np.concatenate((mapped_X, tem_X), axis=1)
    return mapped_X
