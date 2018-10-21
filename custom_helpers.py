# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def standardize_test(x_test, mean, std):
    """Standardize the values of testing_x depending on the values of mean and std of the training x vector"""
    new_x = x_test.copy()
    new_x = (new_x - mean) / std
    return new_x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def remove_invalid(x):
    """Replaces the invalid values -999 by the mean of the entire column"""
    x_new = []
    #first replace the values of -999 with nan
    x_mod = np.where(x == -999, np.nan, x)
    #then obtain the mean of the column ignoring the nan values
    mean = np.nanmean(x_mod, axis=0)
    #now we search the -999 in the original data and replace it with the mean of the column
    for i in range(x.shape[1]):
        x_new.append(np.where(x[:, i] == -999, mean[i], x[:, i]))
    return np.array(x_new).T

def remove_outliers(x):
    """Removes with IQR method, multiplying the IQR by 1.5"""
    data_clean = np.zeros((x.shape))
    for i in range(x.shape[1]):
        col = x[:, i]
        data_copy = np.array(col)
        counts = np.bincount(np.absolute(data_copy.astype(int)))
        replace_most_frecuent = np.argmax(counts)
        upper_quartile = np.percentile(data_copy, 75)
        lower_quartile = np.percentile(data_copy, 25)
        IQR = upper_quartile - lower_quartile
        valid_data = (lower_quartile - IQR*1.5, upper_quartile + IQR*1.5)
        j = 0
        for y in data_copy.tolist():
            if y >= valid_data[0] and y <= valid_data[1]:
                data_clean[j][i] = y
                j = j + 1
            else:
                data_clean[j][i] = replace_most_frecuent
                j = j + 1
    return data_clean

def compute_accuracy(y_pred, y):
    """Computes accuracy of a prediction"""
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy
