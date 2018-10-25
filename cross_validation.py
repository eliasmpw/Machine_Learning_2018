# -*- coding: utf-8 -*-
"""HYPERPARAMETERS TUNING AND CROSS VALIDATION"""

import numpy as np
from methods_implementation import *
from proj1_helpers import *
from custom_helpers import *
import matplotlib.pyplot as plt


#Function to split the data for cross validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

# -CROSS VALIDATION FOR LEAST SQUARES - #
"""We will make the cross validation for least squares """
def least_squares_cross_validation(y, x, k_indices, k, degree):
    "This function will compute the least squares for a specific degree"
    mses_tr = []
    mses_te = []
    for i in range(k):
        newk_index = np.delete(k_indices, i, 0)
        indices_train = newk_index.ravel()
        indices_test = k_indices[i]

        # Train data at each iteration "i" of the loop
        x_train = x[indices_train]
        y_train = y[indices_train]

        # Validate the data at each iteration "i" of the loop
        x_test = x[indices_test]
        y_test = y[indices_test]

        # Building Polynomial base with degree passed
        x_train_poly = build_poly(x_train, degree)
        x_test_poly = build_poly(x_test, degree)

        # Standardizing data
        std_training_x, a, b = standardize(x_train_poly)
        std_testing_x, c, d = standardize(x_test_poly)

        # Getting matrix tX, adding offset value, entire colum of ones[1]
        training_tx = np.c_[np.ones(len(y_train)), std_training_x]
        testing_tx = np.c_[np.ones(len(y_test)), std_testing_x]

        # Use Gradient descendt to find W
        weight_LS, loss_tr_LS = least_squares(y_train, training_tx)
        loss_te_LS = np.sqrt(2 * compute_mse(y_test, testing_tx, weight_LS))
        mses_tr.append(loss_tr_LS)
        mses_te.append(loss_te_LS)

    loss_tr = np.mean(mses_tr)
    loss_te = np.mean(mses_te)
    return loss_tr, loss_te


def least_squares_cross_validation_degree(ytrain, xtrain,k_fold):
    "This function will compute the least squares for a range of degrees and find the best degree"
    seed = 1

    # Indices of each of the groups, each group is 50000
    k_indices = build_k_indices(ytrain, k_fold, seed)

    # Defining possible values for the degree of the polynomial
    degree_range = np.arange(1, 7)
    train_losses = np.zeros(len(degree_range))
    test_losses = np.zeros(len(degree_range))

    # Preparing data for cross validation
    ytrain_cross_validation = ytrain.copy()
    ytrain_cross_validation[np.where(ytrain_cross_validation==-1)] = 0
    xtrain=remove_invalid(xtrain)
    for ind_degree, degree in enumerate(degree_range):
        loss_tr, loss_te = least_squares_cross_validation(ytrain_cross_validation, xtrain, k_indices, k_fold, degree)
        print("For the Degree: %d , The LOSS is : %f" %(degree, loss_te))
        train_losses[ind_degree] = loss_tr
        test_losses[ind_degree] = loss_te
    print("The Least Square Cross Validation finished!!")
    test_losses_abs = np.absolute(test_losses)
    best_value = np.unravel_index(np.argmin(test_losses_abs), test_losses.shape)
    print("The best degrees are: ", degree_range[best_value])
    return degree_range[best_value]



# - CROSS VALIDATION FOR RIDGE REGRESSION - #
def cross_validation_ridge_regression(y, x, k_indices, k, lambda_, degree):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for groupIndex, group in enumerate(k_indices):
        for k_index in group:
            if groupIndex == k:
                x_test.append(x[k_index])
                y_test.append(y[k_index])
            else:
                x_train.append(x[k_index])
                y_train.append(y[k_index])

    # form data with polynomial degree: DONE
    tx_train_poly = build_poly(x_train, degree)
    tx_test_poly = build_poly(x_test, degree)

    # Standardizing data
    std_training_x, h, i = standardize(tx_train_poly)
    std_testing_x, j, k  = standardize(tx_test_poly)
    # ***************************************************
    training_tx = np.c_[np.ones(len(y_train)), std_training_x]
    testing_tx = np.c_[np.ones(len(y_test)), std_testing_x]
    # ridge regression: DONE
    weight,loss_tr = ridge_regression(y_train, training_tx, lambda_)

    # calculate the loss for train and test data: DONE
    loss_te = np.sqrt(2 * compute_mse(y_test, testing_tx, weight))
    # ***************************************************
    return loss_tr, loss_te


def ridge_regression_cross_validation_lambdas(ytrain, xtrain, degree,k_fold):
    seed = 1
    lambda_range = np.arange(-0.005, 0.1, 0.005)
    # split data in k fold
    k_indices = build_k_indices(ytrain, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation: DONE
    for lambda_ in lambda_range:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for index in range(k_fold):
            loss_tr, loss_te = cross_validation_ridge_regression(ytrain, xtrain, k_indices, index, lambda_, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        print("For the Lambda: %f , The LOSS is : %f" %(lambda_,loss_te))
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    # ***************************************************
    print("The Ridge Regression Cross Validation finished!!")
    #cross_validation_visualization(lambda_range, train_losses, test_losses)
    rmse_te_abs = np.absolute(rmse_te)
    best_value = np.unravel_index(np.argmin(rmse_te_abs), rmse_te_abs.shape)
    print("Best lambda is :", lambda_range[best_value])
    return lambda_range[best_value]

def ridge_regression_cross_validation_lambdas_plot(ytrain, xtrain, degree,k_fold):
    seed = 1
    lambda_range = np.arange(-0.005, 0.1, 0.005)
    # split data in k fold
    k_indices = build_k_indices(ytrain, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation: DONE
    for lambda_ in lambda_range:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for index in range(k_fold):
            loss_tr, loss_te = cross_validation_ridge_regression(ytrain, xtrain, k_indices, index, lambda_, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        print("For the Lambda: %f , The LOSS is : %f" %(lambda_,loss_te))
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    # ***************************************************
    print("The Ridge Regression Cross Validation finished!!")
    #cross_validation_visualization(lambda_range, train_losses, test_losses)
    rmse_te_abs = np.absolute(rmse_te)
    best_value = np.unravel_index(np.argmin(rmse_te_abs), rmse_te_abs.shape)
    print("Best lambda is :", lambda_range[best_value])
    return lambda_range[best_value], rmse_tr, rmse_te
