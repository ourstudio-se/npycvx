import numpy as np
import npycvx

im = 10
M = np.array([
    ##b  a  b  c  x   y  t
    [-4,-2, 1, 1,-2, -2, 0],
    [ 5, 0, 0, 0, 5,  0, 1],
    [-2, 0, 0, 0, 0, 10,-1],

    [  0, 0, 0, 0, 0, 0, 1],
    [-10, 0, 0, 0, 0, 0,-1],

    # [ 3, 0, 0, 0, 0, 0, 1]
])
Aub = M[:,1:]
bub = M.T[0]
obj = -1 * np.ones(Aub.shape[1], dtype=np.int)
obj[0]  = 6
obj[-1] = 0

vrs = np.array(list("axytmn"))
bool_vrs = np.array([0,1,2,3,4])
int_vrs = np.array([5])
status, solution = npycvx.solve_lp(
    *npycvx.convert_numpy(
        Aub, 
        bub,
        int_vrs,
        bool_vrs 
    ),
    minimize=False,
    obj=obj
)
print(M)