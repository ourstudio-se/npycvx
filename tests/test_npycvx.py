import functools
import numpy as np
import npycvx

def test_solve_linear_program():
    aub = np.array([
        [-1, 1, 1],
        [-2,-1,-1]
    ])
    bub = np.array([0,-3])
    solve_part_fn = functools.partial(
        npycvx.solve_lp, 
        *npycvx.convert_numpy(aub, bub), 
        False
    )
    objs = np.array([
        [ 0, 0, 0],
        [ 1, 1, 1],
        [-1,-1,-1],
        [ 1, 0, 1],
    ])
    actual = list(map(solve_part_fn, objs))
    expected = [
        ('optimal', np.array([0,0,0])),
        ('optimal', np.array([1,1,0])),
        ('optimal', np.array([0,0,0])),
        ('optimal', np.array([1,0,1])),
    ]
    for s0, v0 in actual:
        assert any((s0==s1 and (v0==v1).all() for s1, v1 in expected))
