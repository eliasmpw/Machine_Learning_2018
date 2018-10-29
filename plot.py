# -*- coding: utf-8 -*-
"""plot helpers for cross validation."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization_lambda(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""

    plt.plot(lambds, mse_tr, marker="x", color='b', label='train error')
    plt.plot(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig("cross_validation_lambda", bbox_inches="tight")

def cross_validation_visualization_degree(degrees, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""

    plt.plot(degrees, mse_tr, marker="x", color='b', label='train error')
    plt.plot(degrees, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=0)
    plt.grid(True)
    plt.ylim((0.6, 1))
    plt.xticks(degrees)
    plt.savefig("cross_validation_degree", bbox_inches="tight")
