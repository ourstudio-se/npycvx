import numpy as np
import scipy.optimize
import nvxopt

M = np.array([
    # a b c d x y z
    [ 0, -1, -1,  0, -1,  1,  0, -2],
    [ 0, -1,  0, -1, -1,  1,  0, -2],
    [-1,  0, -1,  0, -1, -1,  0, -3],
    [-1,  0,  0, -1, -1, -1,  0, -3],
    [-1,  0, -1,  0,  0,  0,  1, -1],
    [-1,  0,  0, -1,  0,  0,  1, -1],
    [ 0, -1,  0, -1,  0,  0,  1, -1]
])
Aub = -1 * M[:,:-1]
bub = -1 * M.T[-1]
obj = -1 * np.ones(Aub.shape[1])

_A, _b, _,_ = scipy.optimize._remove_redundancy._remove_redundancy_pivot_dense(Aub, bub)

status, solution = nvxopt.solve_boolean_lp(obj, Aub, bub)
1