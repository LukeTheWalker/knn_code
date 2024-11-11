from KNNClassifier import KNNClassifier
import numpy as np
from numba.cuda import pinned_array_like 
import sys
import time

start = time.time()

# Example with random data
if len(sys.argv) < 4:
    print('Usage: python test_knn.py <rows> <cols> <k>')
    sys.exit(1)

# Example with random data
rows = int(sys.argv[1])
cols = int(sys.argv[2])
k = int(sys.argv[3])

np.random.seed(699)
X_train_pageable = np.random.rand(rows*cols).reshape((rows,cols))
y_train_pageable = np.random.randint(2, size=rows)

X_train = pinned_array_like(X_train_pageable)
y_train = pinned_array_like(y_train_pageable)

X_train[:] = X_train_pageable
y_train[:] = y_train_pageable

# print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

knn = KNNClassifier(k=k)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
X_test_pageable = np.random.randint(rows, size=test_size)
X_test = pinned_array_like(X_test_pageable)
X_test[:] = X_test_pageable

# Generate Predictions
predictions = knn.predict(X_train[X_test])
#print(f'Prediction {predictions}')
#print(f'Label      {y_train[X_test]}')
# Calculate the number of equal elements
# print(f'correct {np.sum(y_train[X_test] == predictions)}')

end = time.time()
print(f'{rows},{end-start}')