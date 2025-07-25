"""
This module provides a function for a label flipping attack.
"""
import random
import numpy as np


def label_flipping_attack(X: np.ndarray, y: np.ndarray,
                          n_samples: int):
    if n_samples >= X.shape[0]:
        raise ValueError("Too many samples to be poisoned")

    idx = random.sample(list(range(X.shape[0])), k=n_samples)
    y_ = np.copy(y)
    y_[idx] = 1 - y_[idx]

    return X, y_
