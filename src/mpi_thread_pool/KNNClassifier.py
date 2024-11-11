import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance2(self, x1, x2):
        diff = (x1 - x2)
        sqr_diff = diff ** 2
        sqr_diff_sum = np.sum(sqr_diff)
        return np.sqrt(sqr_diff_sum)

    def euclidean_distance(self, x1, X_train):
        return np.sqrt(((x1 - X_train) ** 2).sum(axis=1))

    def predict(self, X):
        with ThreadPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 16))) as executor:
            y_pred = list(executor.map(self._predict, X))

        return np.array(y_pred)

    def _predict(self, x):
        distances = self.euclidean_distance(x, self.X_train)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
