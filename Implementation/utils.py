import itertools
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def get_f_dist(desc, epsilon=1e-3):
    if desc == "l0":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf) < epsilon)
    elif desc == "l1":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf))
    elif desc == "l2":
        return lambda x_orig, x_cf: np.sum(np.square(x_orig - x_cf))
    else:
        raise ValueError(f"Unknown distance function '{desc}'")

class SvcWrapper():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X): # Not provided by LinearSVC
        y_pred_proba = np.zeros((X.shape[0],2))
        y_pred = self.model.predict(X)
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred_proba[i,1] = 1
            else:
                y_pred_proba[i,0] = 1

        return y_pred_proba


def plot_boxplot(X):
    plt.figure()
    plt.boxplot(X)
    plt.show()
