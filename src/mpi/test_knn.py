from mpi4py import MPI
from KNNClassifier import KNNClassifier
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) < 4:
    print('Usage: python test_knn.py <rows> <cols> <k>')
    sys.exit(1)

# Example with random data
rows = int(sys.argv[1])
cols = int(sys.argv[2])
k = int(sys.argv[3])

test_size = 1000

if rank == 0:
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

#print(f'Rank {rank} - Start index: {start_idx} - End index: {end_idx}')

# Each process initializes the classifier and fits the model
knn = KNNClassifier(k=k)
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