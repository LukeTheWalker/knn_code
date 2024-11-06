from KNNClassifier import KNNClassifier
import numpy as np
import time
import sys

start = time.time()

if len(sys.argv) < 4:
    print('Usage: python test_knn.py <rows> <cols> <k>')
    sys.exit(1)

# Example with random data
rows = int(sys.argv[1])
cols = int(sys.argv[2])
k = int(sys.argv[3])
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)

knn = KNNClassifier(k=k)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

# Generate Predictions
predictions = knn.predict(X_train[X_test])
#print(f'Prediction {predictions}')
#print(f'Label      {y_train[X_test]}')
# Calculate the number of equal elements
# print(f'correct {np.sum(y_train[X_test] == predictions)}')

end = time.time()
print(f'{end-start}')

