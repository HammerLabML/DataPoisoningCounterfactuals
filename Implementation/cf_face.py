from copy import deepcopy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse.csgraph import dijkstra


def fit_kde_to_data(X):
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
    kde.fit(X)

    return lambda x: np.exp(kde.score_samples(x))


# Implementation of FACE Counterfactual Generator -- see https://dl.acm.org/doi/pdf/10.1145/3375627.3375850
# ATTENTION: We use an implementation similar to the one in CARLA -- see https://github.com/carla-recourse/CARLA/blob/main/carla/recourse_methods/catalog/face/library/face_method.py
class FACE():
    def __init__(self, X, y, clf, p=lambda x: 1., tp=.8, td=.2, graph_desc="k-nn", graph_options={"k": 50, "r": .5}, norm=2):
        self.X = X
        self.y = y
        self.tp = tp
        self.td = td
        self.norm = norm
        self.clf = clf
        self.graph_options = graph_options

        if p is not None:
            self.p = p
        else:       # Default: use KDE
            kde = KernelDensity()
            kde.fit(self.X)
            self.p = lambda x: np.exp(kde.score_samples(x.reshape(1,-1)))

        self.W = None
        if graph_desc == "k-nn":
            self.W = kneighbors_graph(self.X, n_neighbors=self.graph_options["k"], mode="distance", p=self.norm)
        elif graph_desc == "eps-graph":
            self.W = radius_neighbors_graph(self.X, radius=self.graph_options["r"], mode="distance", p=self.norm)
        else:
            raise ValueError(f"Unknown graph '{graph_desc}'")
                       
        # Generate candidate targets
        self.I_CT = []
        y_pred = self.clf.predict(self.X)
        y_pred_proba = self.clf.predict_proba(self.X)
        #densities = self.p(X)  # TODO: Fix density estimator
        densities = [self.p(x) for x in self.X]
        for i in range(self.X.shape[0]):
            if densities[i] >= self.td and y_pred_proba[i, self.y[i]] >= self.tp and y_pred[i] == y[i]:  # Only consider correctly classified samples!
                self.I_CT.append(i)

    def compute_cf_path(self, x_idx, y_target, min_length=0):
        # Create list of suitable targets
        target_idx = list(filter(lambda i: self.y[i] == y_target, self.I_CT))

        # Compute shortest path from original sample to all suitable targets
        dist_matrix, pred_array = dijkstra(csgraph=self.W, indices=x_idx, directed=True, return_predecessors=True)

        dist_matrix_idx = dist_matrix[target_idx] > min_length  # Counterfactual path must be greater or equal than some given minimum length
        dist_matrix_ = dist_matrix[target_idx][dist_matrix_idx]
        target_idx_ = np.array(target_idx)[dist_matrix_idx]

        cf_idx = target_idx_[np.argmin(dist_matrix_)]

        # Get final path and compute distance of each "hop"
        cur_node_idx = cf_idx
        cfpath_samples_idx = [cf_idx]
        while(cur_node_idx != x_idx):
            cur_node_idx = pred_array[cur_node_idx]
            cfpath_samples_idx.append(cur_node_idx)
        cfpath_samples_idx.reverse()

        cfpath_samples_labels = [self.y[idx] for idx in cfpath_samples_idx]

        cfpath_dists = []
        for i in range(len(cfpath_samples_idx)-1):
            cfpath_dists.append(self.W[cfpath_samples_idx[i], cfpath_samples_idx[i+1]])

        return cfpath_samples_idx, [self.X[idx,:] for idx in cfpath_samples_idx], cfpath_samples_labels, cfpath_dists



class FaceExplainer():
    def __init__(self, clf, X_train, X, y):
        self.clf = clf
        self.kde = fit_kde_to_data(X_train)
        self.X = X
        self.y = y
    
    def compute_counterfactual(self, x, y_orig, y_target):
        exp = FACE(np.concatenate((x.reshape(1, -1), self.X), axis=0), np.concatenate(([y_orig], self.y)), self.clf)#, p=self.kde)  # TODO: Fix KDE
        cf_idx, _, _, _ =  exp.compute_cf_path(0, y_target)
        return [self.X[cf_idx[-1], :]]
