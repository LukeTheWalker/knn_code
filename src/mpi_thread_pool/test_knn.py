from mpi4py import MPI
from KNNClassifier import KNNClassifier
import numpy as np
import os

import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
rows = int(os.getenv("ROWS", "100000"))
cols = 500
test_size = 1000

# Root process generates training and test data
if rank == 0:
    start = time.time()
    np.random.seed(699)
    X_train = np.random.rand(rows * cols).reshape((rows, cols))
    y_train = np.random.randint(2, size=rows)
    X_test_indices = np.random.randint(rows, size=test_size)
else:
    X_train = None
    y_train = None
    X_test_indices = None

# Broadcast training data and labels to all processes
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
X_test_indices = comm.bcast(X_test_indices, root=0)

# Distribute test data indices among MPI processes
local_test_size = test_size // size
remainder = test_size % size

# Adjust the range for each process to handle the remainder
if rank < remainder:
    start_idx = rank * (local_test_size + 1)
    end_idx = start_idx + local_test_size + 1
else:
    start_idx = rank * local_test_size + remainder
    end_idx = start_idx + local_test_size

local_X_test_indices = X_test_indices[start_idx:end_idx]

# Each process initializes the classifier and fits the model
knn = KNNClassifier(k=2)
knn.fit(X_train, y_train)

# Generate predictions
local_predictions = knn.predict(X_train[local_X_test_indices])

# Gather all predictions on the root process using Gatherv
all_predictions = None
sendcounts = comm.gather(len(local_predictions), root=0)

if rank == 0:
    all_predictions = np.empty(test_size, dtype=int)

# Gather variable-sized predictions
comm.Gatherv(local_predictions, (all_predictions, sendcounts), root=0)

# Root process calculates and prints the accuracy
if rank == 0:
    correct = np.sum(all_predictions == y_train[X_test_indices])
    print(f'Correct Predictions: {correct}')
    print(f'Accuracy: {correct / test_size * 100:.2f}%')
    end = time.time()
    elapsed_time = end - start
    print(f'elapsed time:{elapsed_time:.6f}')
