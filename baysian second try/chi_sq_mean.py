import numpy as np


def chi_sq_mean(rv, errors):
    return np.sum((rv - np.mean(rv))**2 / errors**2)
