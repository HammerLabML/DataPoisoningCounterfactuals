import numpy as np


class MemoryCounterfactual():
    def __init__(self, X, norm=1):
        self.X = X
        self.dist = self.__build_norm(norm)

    def __build_norm(self, norm_desc):
        return lambda x: np.linalg.norm(x, ord=norm_desc)

    def compute_counterfactual(self, x_orig):
        X_diff = self.X - x_orig
        dist = [self.dist(X_diff[i,:].flatten()) for i in range(X_diff.shape[0])]
        idx = np.argmin(dist)

        return self.X[idx,:]
