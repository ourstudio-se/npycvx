import cvxopt
import cvxopt.glpk
import contextlib
import ctypes
import os, sys
import numpy as np

@contextlib.contextmanager
def redirect_stdout2devnull():

    libc = ctypes.CDLL(None)
    try:
        c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
    except ValueError as e:
        # If 'stdout' doesn't work try this.
        # On Mac OS X the symbol for standard output in C is '__stdoutp' which is needed for mac users to run this locally
        c_stdout = ctypes.c_void_p.in_dll(libc, '__stdoutp')

    devnull = open(os.devnull, 'wb')

    try:
        stdout_fd = sys.stdout.fileno()
    except ValueError:
        redirect = False
    else:
        redirect = True

        # Flush both the C-level stdout and sys.stdout to print everything that's currently in the buffers
        libc.fflush(c_stdout)
        sys.stdout.flush()

        devnull_fd = devnull.fileno()
        saved_stdout_fd = os.dup(stdout_fd)

        # Make the file descriptor of stdout point to the same file as devnull
        os.dup2(devnull_fd, stdout_fd)

    yield

    if redirect:
        # Change back so that the file descriptor of stdout points to the original file
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)

cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
def solve_boolean_lp(
    obj: np.ndarray,
    aub: np.ndarray,
    bub: np.ndarray,
    aeq: np.ndarray=None,
    beq: np.ndarray=None,
    minimize: bool=True):
    """
        Solve a BLP (boolean linear programming) problem of the type

        Min obj * x           or    Max obj * x
        s.t. aub * x ≥ bub          s.t. aub * x ≥ bub
             aeq * x = beq               aeq * x = beq
        (^default)

        Parameters
        obj: The objective function coefficients as an ndarray with shape (n,)
        aub: Constraint matrix as an ndarray with shape (m, n)
        bub: Support vector as an ndarray with shape (m,)
        aeq: Constraint matrix as an ndarray with shape (p, n)
        beq: Support vector as an ndarray with shape (p,)
        minimize: If True use the minimization BLP, otherwise use maximization BLP

        Returns
        status: The status returned by the GLPK ILP solver (see help(cvxopt.glpk.ilp) for info)
        x: If the status is 'optimal', 'feasible' or 'undefined' then this will be a solution vector as an ndarray
    """

    if len(obj.shape) != 1:
        raise ValueError("'obj' must be a one-dimensional array of shape: (n,)")

    dim = obj.shape[0]

    def validate_constraint_format(mat: np.ndarray, sup: np.ndarray) -> bool:
        if (mat is not None) or (sup is not None):
            if not (isinstance(mat, np.ndarray) and isinstance(sup, np.ndarray)):
                return False

            if len(mat.shape) != 2 or len(sup.shape) != 1:
                return False

            if mat.shape[1] != dim or mat.shape[0] != sup.shape[0]:
                return False
        return True

    if not validate_constraint_format(aub, bub):
        raise ValueError("'aub' must be a two-dimensional ndarray of shape (m, n) and 'bub' must be a one-dimensional ndarray of shape (m,) where n is the length of 'obj'")
    if not validate_constraint_format(aeq, beq):
        raise ValueError("'aeq' must be a two-dimensional ndarray of shape (m, n) and 'beq' must be a one-dimensional ndarray of shape (m,) where n is the length of 'obj'")

    if minimize:
        # Minimize (default)
        c = cvxopt.matrix(obj.astype(np.float).tolist())
    else:
        # Maximize -> swap sign on the objective
        c = cvxopt.matrix((obj * -1).astype(np.float).tolist())

    # Constraint format is >=
    # Swap sign on coefficients in constraint matrix and support vector
    G = cvxopt.matrix((aub * -1).astype(np.float).T.tolist()) if not aub is None else None
    h = cvxopt.matrix((bub * -1).astype(np.float).tolist()) if not bub is None else None

    A = cvxopt.matrix(aeq.astype(np.float).T.tolist()) if not aeq is None else None
    b = cvxopt.matrix(beq.astype(np.float).tolist()) if not beq is None else None
    I = set()  # No variables should be integer ...
    B = set(i for i in range(dim))  # ... all should be boolean (0 or 1)

    with redirect_stdout2devnull():
        status, x = cvxopt.glpk.ilp(c, G, h, A, b, I, B)

    if status in ['optimal', 'feasible', 'undefined']:
        x = np.array(x).flatten().astype(np.int16)

    return (status, x)

def convert_numpy(
    aub: np.ndarray,
    bub: np.ndarray,
    aeq: np.ndarray=None,
    beq: np.ndarray=None,
):
    """
        Convert numpy matrices into a BLP (boolean linear programming) problem of the type

        Min obj * x           or    Max obj * x
        s.t. aub * x ≥ bub          s.t. aub * x ≥ bub
             aeq * x = beq               aeq * x = beq
        (^default)

        Parameters
        aub: Constraint matrix as an ndarray with shape (m, n)
        bub: Support vector as an ndarray with shape (m,)
        aeq: Constraint matrix as an ndarray with shape (p, n)
        beq: Support vector as an ndarray with shape (p,)

        Returns
            A loaded linprog func with constraints.
            x = load_boolean_lp(...)
            then x(obj) = sol
    """

    dim = aub.shape[1]
    def validate_constraint_format(mat: np.ndarray, sup: np.ndarray) -> bool:
        if (mat is not None) or (sup is not None):
            if not (isinstance(mat, np.ndarray) and isinstance(sup, np.ndarray)):
                return False

            if len(mat.shape) != 2 or len(sup.shape) != 1:
                return False

            if mat.shape[1] != dim or mat.shape[0] != sup.shape[0]:
                return False
        return True

    if not validate_constraint_format(aub, bub):
        raise ValueError("'aub' must be a two-dimensional ndarray of shape (m, n) and 'bub' must be a one-dimensional ndarray of shape (m,) where n is the length of 'obj'")
    if not validate_constraint_format(aeq, beq):
        raise ValueError("'aeq' must be a two-dimensional ndarray of shape (m, n) and 'beq' must be a one-dimensional ndarray of shape (m,) where n is the length of 'obj'")

    # Constraint format is >=
    # Swap sign on coefficients in constraint matrix and support vector
    G = cvxopt.matrix((aub * -1).astype(np.float).T.tolist()) if not aub is None else None
    h = cvxopt.matrix((bub * -1).astype(np.float).tolist()) if not bub is None else None

    A = cvxopt.matrix(aeq.astype(np.float).T.tolist()) if not aeq is None else None
    b = cvxopt.matrix(beq.astype(np.float).tolist()) if not beq is None else None
    I = set()  # No variables should be integer ...
    B = set(i for i in range(dim))  # ... all should be boolean (0 or 1)

    return G, h, A, b, I, B

def solve_lp(G: cvxopt.matrix, h: cvxopt.matrix, A: cvxopt.matrix, b: cvxopt.matrix, I: set, B: set, minimize: bool, obj: np.ndarray) -> tuple:

    """
        Solves linear programming problem where

            G:          cvxopt matrix (Aub)
            h:          cvxopt matrix (bub)
            A:          cvxopt matrix (Aeq)
            b:          cvxopt matrix (beq)
            I:          set of integer variables
            B:          set of boolean variables
            minimize:   bool indicating if minimize (true) or maximizing (false) objective
            obj:        numpy array/vector
        as such

            Min obj * x           or    Max obj * x
            s.t. aub * x ≥ bub          s.t. aub * x ≥ bub
                aeq * x = beq               aeq * x = beq
            (^default)

        Tip:
            This function is useful when optimizing over many
            objectives while the constraints stays the same.
            Here's an example of how to do: 
                Input:
                    aub = np.array([
                        [-1, 1, 1],
                        [-2,-1,-1]
                    ])
                    bub = np.array([0,-3])
                    part_args = convert_numpy(aub, bub)
                    solve_part_fn = functools.partial(solve_lp, *part_args, True)
                    objs = np.array([
                        [ 0, 0, 0],
                        [ 1, 1, 1],
                        [-1,-1,-1],
                        [ 1, 0, 1],
                    ])
                    solutions = list(map(solve_part_fn, objs))

                Output:
                    

        Return:
            tuple (
                status = {'optimal', 'feasible', 'undefined'}, 
                numpy.ndarray
            )
    """

    if minimize:
        # Minimize (default)
        c = cvxopt.matrix(obj.astype(np.float).tolist())
    else:
        # Maximize -> swap sign on the objective
        c = cvxopt.matrix((obj * -1).astype(np.float).tolist())

    with redirect_stdout2devnull():
        status, x = cvxopt.glpk.ilp(c, G, h, A, b, I, B)

    if status in ['optimal', 'feasible', 'undefined']:
        x = np.array(x).flatten().astype(np.int16)

    return (status, x)
