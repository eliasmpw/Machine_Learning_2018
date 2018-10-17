# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


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


def remove_outliers_max(data):
    """Replaces invalid data=-999 with the mean of the column"""
    dataset_clean = []
    dataset_nan = np.where(data == -999, np.nan, data)
    mean = np.nanmean(dataset_nan, axis=0)

    for i in range(data.shape[1]):
        dataset_clean.append(np.where(data[:, i] == -999, mean[i], data[:, i]))

    return np.array(dataset_clean).T


def remove_outliers(data):
    """Replaces the ouliers with the most frecuent number of the colum"""

    dataset_clean = np.zeros((data.shape))
    for i in range(data.shape[1]):

        col = data[:, i]
        array = np.array(col)
        counts = np.bincount(np.absolute(array.astype(int)))
        y1 = np.argmax(counts)

        upper_quartile = np.percentile(array, 75)
        lower_quartile = np.percentile(array, 25)
        IQR = (upper_quartile - lower_quartile) 
        quartileSet = (lower_quartile - IQR*1.5, upper_quartile + IQR*1.5)
        j = 0
        
        for y in array.tolist():
            if y >= quartileSet[0] and y <= quartileSet[1]:
                dataset_clean[j][i] = y
                j=j+1
            else:
                dataset_clean[j][i] = y1
                j=j+1

    return dataset_clean



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
            
            
            
