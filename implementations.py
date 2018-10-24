# -*- coding: utf-8 -*-
"""Contains the 6 ML methods that were required to be implemented."""
import csv
import numpy as np


# ----------------------- Least Squares using Gradient Descent ---------------------------
def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for _ in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        # compute loss
        loss = compute_mse(y, tx, w)
    return w, loss


# ----------------------- Least Squares using Stochastic Gradient Descent ---------------------------
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


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

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w
    batch_size = 1

    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
    return w, loss


# ----------------------- Least Squares ---------------------------


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Ridge Regression ---------------------------
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Logistic Regression ---------------------------
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def calculate_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using full gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        # compute loss, gradient
        grad = calculate_log_gradient(y, tx, w)
        loss = calculate_log_loss(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    return w, loss


# ----------------------- Regularized Logistic Regression ---------------------------
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement regularized logistic regression using full gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        # compute loss, gradient
        gradient = calculate_log_gradient(y, tx, w) + 2 * lambda_ * w
        loss = calculate_log_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        # gradient w by descent update
        w = w - gamma * grad
    return w, loss


# ----------------------- Prediction functions --------------------------------------
def predict_linreg(w, tx):
    """Make prediction for a linear model"""
    pred = tx.dot(w)
    return pred


def predict_logreg(w, tx):
    """Make prediction for a logistic model"""
    pred = sigmoid(tx.dot(w))
    return pred