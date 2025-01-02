import numpy as np
from sklearn.preprocessing import Normalizer

X_test = [
    [-4., 0.],
    [-1., 3.]
]

nz = Normalizer(norm='l2')
Y_test = nz.transform(X_test)

print(X_test)
print(Y_test)

mod = np.linalg.norm(np.array(Y_test[1]))

print(np.arccos(np.dot(Y_test[0], Y_test[1])))