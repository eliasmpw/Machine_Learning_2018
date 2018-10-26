# -*- coding: utf-8 -*-
"""Contains the 6 ML methods that were required to be implemented."""
import csv
import numpy as np
from custom_helpers import *

# ----------------------- Least Squares using Gradient Descent ---------------------------
def compute_mse(y, tx, w):
    """Compute the loss by mse.
    Args:
        y: y values.
        tx: transposed x values.
        w: weight.
    Returns:
        The calculated MSE.
    """
    pred = np.squeeze(tx.dot(w))
    e = y - pred
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient.
    Args:
        y: y values.
        tx: transposed x values.
        w: weight.
    Returns:
        The calculated gradient.
    """
    pred = np.squeeze(tx.dot(w))
    err = y - pred
    gradient = -tx.T.dot(err) / len(err)
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
    Args:
        y: y values.
        tx: transposed x values.
        initial_w: initial weight.
        max_iters: number of iterations.
        gamma: the gamma to use.
    Returns:
        w: weight result.
        loss: loss result.
    """
    w = initial_w
    print_every = 50
    cumulative_loss = 0

    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * gradient
        # compute loss
        loss = compute_mse(y, tx, w)
        cumulative_loss += loss

        if (n_iter % print_every==0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ', str(cumulative_loss / print_every))
            cumulative_loss = 0;

    return w, loss


# ----------------------- Least Squares using Stochastic Gradient Descent ---------------------------
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    Args:
        y: y values.
        tx: transposed x values.
        w: initial weight.
    Returns:
        grad: gradient result.
        err: error result.
    """
    err = y - tx.dot(w)
    gradient = -tx.T.dot(err) / len(err)
    return gradient, err


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

    """Stochastic gradient descent.
    Args:
        y: y values.
        tx: transposed x values.
        initial_w: initial weight.
        max_iters: number of iterations.
        gamma: the gamma to use.
    Returns:
        w: weight result.
        loss: loss result.
    """
    # Define parameters to store w and loss
    w = initial_w
    batch_size = 1
    print_every = 50
    cumulative_loss = 0

    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            cumulative_loss += loss

        if (n_iter % print_every==0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ', str(cumulative_loss / print_every))
            cumulative_loss = 0;

    return w, loss


# ----------------------- Least Squares ---------------------------


def least_squares(y, tx):
    """calculate the least squares solution.
    Args:
        y: y values.
        tx: transposed x values.
    Returns:
        w: weight result.
        loss: loss result.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Ridge Regression ---------------------------
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    Args:
        y: y values.
        tx: transposed x values.
        lambda_: the lambda value to use.
    Returns:
        w: weight result.
        loss: loss result.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return w, loss


# ----------------------- Logistic Regression ---------------------------
def sigmoid(t):
    """Apply sigmoid function on t.
    Args:
        t: value to use.
    Returns:
        Calculated sigmoid
    """
    return 1.0 / (1.0 + np.exp(-t))


def calculate_log_loss(y, tx, w):
    """Compute the cost by negative log likelihood.
    Args:
        y: y values.
        tx: transposed x values.
        w: weight.
    Returns:
        Calculated logistic loss
    """
    pred = sigmoid(tx.dot(w))
    
    z = (1 + y) / 2
    loss = (z.T.dot(np.log(pred)) + (1 - z).T.dot(np.log(1 - pred))) / len(y)
    return np.squeeze(- loss)


def calculate_log_gradient(y, tx, w):
    """Compute the gradient of loss.
    Args:
        y: y values.
        tx: transposed x values.
        w: weight.
    Returns:
        Calculated logistic gradient
    """
    #variable change
    z = (1 + y) / 2
    pred = sigmoid(tx.dot(w))   
    grad = tx.T.dot(pred - z)
    return grad


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using SGD.
    Args:
        y: y values.
        tx: transposed x values.
        initial_w: initial weight.
        max_iters: number of iterations.
        gamma: the gamma to use.
    Returns:
        w: weight result.
        loss: loss result.
    """
    batch_size = 1
    print_every = 50
    cumulative_loss = 0

    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute loss, gradient
            grad = calculate_log_gradient(y_batch, tx_batch, w)
            loss = calculate_log_loss(y_batch, tx_batch, w)
            cumulative_loss += loss
            # gradient w by descent update
            w = w - gamma * grad

        if (n_iter % print_every==0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ', str(cumulative_loss / print_every))
            cumulative_loss = 0;
    return w, loss


# ----------------------- Regularized Logistic Regression ---------------------------
def reg_logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, lambda_):
    """implement regularized logistic regression using SGD.
    Args:
        y: y values.
        tx: transposed x values.
        initial_w: initial weight.
        max_iters: number of iterations.
        gamma: the gamma to use.
    Returns:
        w: weight result.
        loss: loss result.
    """
    batch_size = 1
    print_every = 50
    cumulative_loss = 0    
    w = initial_w


    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute loss, gradient
            grad = calculate_log_gradient(y_batch, tx_batch, w) + 2 * lambda_ * w
            loss = calculate_log_loss(y_batch, tx_batch, w) + lambda_ * np.squeeze(w.T.dot(w))
            cumulative_loss += loss
            # gradient w by descent update
            w = w - gamma * grad

        if (n_iter % print_every==0):
            # print average loss for the last print_every iterations
            print('iteration\t', str(n_iter), '\tloss: ', str(cumulative_loss / print_every))
            cumulative_loss = 0;
    return w, loss
