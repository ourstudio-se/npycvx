# NpyCVX
A small library to connect [numpy](https://numpy.org/) and [CVXOPT](https://cvxopt.org/) together and solves all messy conversions in between.

## Install
```bash
pip install npycvx
```

## Example usage
A simple example when maximizing `w^T x` over the same system of linear inequalities.
```python

import numpy as np
import npycvx
import functools # <- built-in python lib... 

# Some dummy data...
A = np.array([
    [-1, 1, 1],
    [-2,-1,-1]
])
b = np.array([0,-3])
objectives = np.array([
    [ 0, 0, 0],
    [ 1, 1, 1],
    [-1,-1,-1],
    [ 1, 0, 1],
])

# Load solve-function with the now converted numpy
# matrices/vectors into cvxopt data type...
solve_part_fn = functools.partial(
    npycvx.solve_lp, 
    *npycvx.convert_numpy(A, b), 
    False
)

# Exectue each objective with solver function
solutions = list(
    map(
        solve_part_fn, 
        objectives
    )
)
```
