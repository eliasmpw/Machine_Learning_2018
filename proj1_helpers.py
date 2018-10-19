# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import *

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """

    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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
