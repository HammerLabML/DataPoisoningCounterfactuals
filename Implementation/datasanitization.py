import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class MyIsloationForest():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.method = IsolationForest()
        self.method.fit(X_train, y_train)

    def predict(self, X, y) -> np.ndarray:
        return self.method.predict(X)


class MyLocalOutlierFactor():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.method = LocalOutlierFactor(novelty=True)
        self.method.fit(X_train, y_train)

    def predict(self, X, y) -> np.ndarray:
        return self.method.predict(X)


class L2Defense():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 threshold: float):
        self.X_train = np.copy(X_train)
        self.y_train = np.copy(y_train)
        self.threshold = threshold

    def __compute_distance(self, X: np.ndarray, y: np.ndarray) -> list:
        centroid_0 = np.mean(self.X_train[self.y_train == 0, :], axis=0)
        centroid_1 = np.mean(self.X_train[self.y_train == 1, :], axis=0)

        dist_0 = np.linalg.norm(X - centroid_0, 2, axis=1)
        dist_1 = np.linalg.norm(X - centroid_1, 2, axis=1)

        y_pred = []
        for y_, d0, d1 in zip(y, dist_0, dist_1):
            if y_ == 0:
                y_pred.append(d0)
            else:
                y_pred.append(d1)

        return y_pred

    def calibrate_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        dists = self.__compute_distance(X, y)
        self.threshold = np.mean(dists) + np.var(dists)

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = np.array([-1 if d >= self.threshold else 1 for d in self.__compute_distance(X, y)])
        return y_pred


class SlabDefense():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 threshold: float):
        self.X_train = np.copy(X_train)
        self.y_train = np.copy(y_train)
        self.threshold = threshold

    def __compute_distance(self, X: np.ndarray, y: np.ndarray) -> list:
        centroid_0 = np.mean(self.X_train[self.y_train == 0, :], axis=0)
        centroid_1 = np.mean(self.X_train[self.y_train == 1, :], axis=0)

        centroid_diff = centroid_0 - centroid_1

        y_pred = []
        for x, y_ in zip(X, y):
            if y_ == 0:
                y_pred.append(np.abs(np.dot(x - centroid_0, centroid_diff)))
            else:
                y_pred.append(np.abs(np.dot(x - centroid_1, centroid_diff)))

        return y_pred

    def calibrate_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        dists = self.__compute_distance(X, y)
        self.threshold = np.mean(dists) + np.var(dists)

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = np.array([-1 if d >= self.threshold else 1 for d in self.__compute_distance(X, y)])
        return y_pred


class KnnDefense():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 k: int, threshold: float):
        self.k = k
        self.threshold = threshold
        self.knn = {0: KDTree(data=X_train[y_train == 0, :], metric="euclidean"),
                    1: KDTree(data=X_train[y_train == 1, :], metric="euclidean")}

    def __compute_distance(self, X: np.ndarray, y: np.ndarray) -> list:
        y_pred = []
        for x, y_ in zip(X, y):
            dist, _ = self.knn[y_].query(x.reshape(1, -1), k=self.k)
            y_pred.append(dist[0][self.k-1])

        return y_pred

    def calibrate_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        dists = self.__compute_distance(X, y)
        self.threshold = np.mean(dists) + np.var(dists)

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = np.array([-1 if d >= self.threshold else 1 for d in self.__compute_distance(X, y)])
        return y_pred
