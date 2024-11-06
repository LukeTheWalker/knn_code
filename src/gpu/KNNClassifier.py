import numpy as np
from numba import cuda, guvectorize, jit, njit
import math
import cupy as cp
from tqdm import tqdm

# CUDA kernel to compute Euclidean distance
@cuda.jit
def euclidean_distance_kernel(x, data, distances):
    i = cuda.grid(1)
    if i < data.shape[0]:
        diff = 0
        for j in range(data.shape[1]):
            diff += (x[j] - data[i][j])**2
        distances[i] = math.sqrt(diff)

class KNNClassifier:
    def __init__(self, k, n_gpus = 4):
        self.k = k
        self.n_gpus = n_gpus

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.d_X_train = []
        self.d_y_train = []
        self.streams = []
        self.n_points_per_gpu = X.shape[0] // self.n_gpus
        for i in range(self.n_gpus):
            cuda.select_device(i)
            self.streams.append(cuda.stream())
            start = i*self.n_points_per_gpu
            end = min((i+1)*self.n_points_per_gpu, X.shape[0])
            self.d_X_train.append(cuda.to_device(X[start:end], stream=self.streams[i]))
            self.d_y_train.append(cuda.to_device(y[start:end], stream=self.streams[i]))

    def predict(self, X):
        # y_pred = [self._predict(x) for x in X]
        y_pred = []
        for x in tqdm(X):
            y_pred.append(self._predict(x))
        return np.array(y_pred)

    def _predict(self, x):
        n_points = self.X_train.shape[0]

        threads_per_block = (256)
        blocks_per_grid = (n_points + (threads_per_block - 1)) // threads_per_block

        d_distances = []
        x_devices = []

        for i in range(self.n_gpus):
            cuda.select_device(i)
            d_distances.append(cuda.device_array(self.d_X_train[i].shape[0], stream=self.streams[i]))
            x_devices.append(cuda.to_device(x, stream=self.streams[i]))

        for i in range(self.n_gpus):
            cuda.select_device(i)
            euclidean_distance_kernel[blocks_per_grid, threads_per_block, self.streams[i]](x_devices[i], self.d_X_train[i], d_distances[i])

        top_k_relative = []
        top_k_distances_relative = []
        for i in range(self.n_gpus):
            cuda.select_device(i)
            top_k_relative.append(cp.argsort(cp.asarray(d_distances[i]))[:self.k].get())
            top_k_distances_relative.append(cp.sort(cp.asarray(d_distances[i]))[:self.k].get())

        top_k = [ i  + self.n_points_per_gpu * j for j, candidate in enumerate(top_k_relative) for i in list(candidate)]

        distances = np.concatenate(top_k_distances_relative)

        k_nearest_small = np.argsort(distances)[:self.k]

        k_nearest = [top_k[i] for i in k_nearest_small]

        k_nearest_labels = self.y_train[k_nearest]

        return np.bincount(k_nearest_labels).argmax()