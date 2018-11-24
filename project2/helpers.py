# -*- coding: utf-8 -*-
"""some helper functions for project 2."""
import csv
import numpy as np
import math

#### ----------------- taken from project 1 implementations ----------------####
def batch_iter(y, tx, batch_size, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        batch_size (int): Size of the batch
        num_batches (int, optional): Defaults to 1. Number of batches
        shuffle (bool, optional): Defaults to True. Shuffle or not
    """

    data_size = len(y)
    num_batches = math.floor(data_size / batch_size)

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


#### --------------- taken from project 1 helpers ---------------------####
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


#### ----------------------- new methods --------------------####
def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / len(labels)
