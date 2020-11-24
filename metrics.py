import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def absolute_errors(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)
