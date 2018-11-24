# -*- coding: utf-8 -*-
"""implementation of logistic regression"""

import numpy as np

from helpers import batch_iter

def sigmoid(t):
    """Apply sigmoid function on t

    Args:
        t (numpy.array): Value to use

    Returns:
        numpy.array: Calculated sigmoid
    """

    return 1.0 / (1.0 + np.exp(-t))


def calculate_log_loss(y, tx, w):
    """Compute the cost by negative log likelihood

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        w (numpy.array): Weight

    Returns:
        numpy.array: Calculated logistic loss
    """

    xtw = (tx.dot(w))
    loss = np.sum(np.log(1.0 + np.exp(xtw)) - np.multiply(y, xtw))

    return loss


def calculate_log_gradient(y, tx, w):
    """Compute the gradient of loss.

    Args:
        y (numpy.array): y values
        tx (numpy.array): Transposed x values
        w (numpy.array): Weight

    Returns:
        numpy.array: Calculated logistic gradient
    """

    sig = sigmoid(tx.dot(w))
    gradient = tx.T.dot(sig - y)

    return gradient


def reg_logistic_regression(y, tx, initial_w, epochs, batch_size, gamma, lambda_, print_every=1):
    """Implement regularized logistic regression using full gradient descent

    Args:
        y (numpy.array): y values (either 1 or -1)
        tx (numpy.array): transposed x values
        initial_w (numpy.array): initial weight.
        epochs (int): number of epochs.
        batch_size (int): batch size.
        gamma(float): the gamma to use.
        lambda(float): the lambda to use.

    Returns:
        (tuple): tuple containing:

            w (numpy.array): Weight result
            loss (numpy.array): Loss result
    """

    w = initial_w

    # change labels from (-1, 1) to (0, 1)
    y_bin = y == 1

    for epoch in range(epochs):
        n_iter = 0
        cumulative_loss = 0
        for y_batch, tx_batch in batch_iter(y_bin, tx, batch_size):
            loss = calculate_log_loss(
                y_batch, tx_batch, w) + lambda_ * np.squeeze(w.T.dot(w))
            grad = calculate_log_gradient(
                y_batch, tx_batch, w) + 2 * lambda_ * w
            w -= gamma * grad
            cumulative_loss += loss
            n_iter += 1
            if (n_iter == print_every):
                # print average loss for the last print_every iterations
                print('epoch\t', str(epoch+1), '\tloss: ',
                    str(cumulative_loss / print_every))
                cumulative_loss = 0
                n_iter = 0
    return w, loss


def predict_logistic_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
