import numpy as np

X = np.array([[0, 1, 2], [3, 4, 5]])
W = np.array([[1, 2, 3],
              [4, 5, 6]])

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
y = np.dot(a, b)

print(y)
