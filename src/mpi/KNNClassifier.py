import numpy as np
from mpi4py import MPI

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, X_train):
        return np.sqrt(((x1 - X_train) ** 2).sum(axis=1))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]

        return np.array(y_pred)

    def _predict(self, x):
        distances = self.euclidean_distance(x, self.X_train)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common