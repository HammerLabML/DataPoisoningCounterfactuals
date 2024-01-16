import numpy as np

from utils import get_f_dist


class MemoryCounterfactual():
    def __init__(self, X, y, f_dist="l1"):
        self.X = X
        self.y = y
        self.dist = f_dist
        if not callable(self.dist):
            self.dist = get_f_dist(f_dist)

    def compute_counterfactual(self, x_orig, y_target):
        mask = self.y == y_target
        X_ = self.X[mask,:]
        
        X_diff = X_ - x_orig
        dist = [self.dist(x_orig, X_[i,:]) for i in range(X_diff.shape[0])]
        idx = np.argmin(dist)

        return X_[idx,:]


class MemoryExplainer():
    def __init__(self, clf, X_train, y_train):
        self.clf = clf

        y_pred = clf.predict(X_train)
        mask = y_pred == y_train    # Limit the feasible set to correctly classified samples
        X = X_train[mask,:]
        y = y_train[mask]
        self.exp = MemoryCounterfactual(X, y)

    def compute_counterfactual(self, x, y_target):
        return [self.exp.compute_counterfactual(x, y_target)]
