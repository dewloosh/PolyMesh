from numba import njit, prange
import numpy as np
from numpy import ndarray
from neumann import flatten2dC

__cache = True


@njit(nogil=True, cache=__cache)
def shp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [
                0.25 * r**2 * s**2
                - 0.25 * r**2 * s
                - 0.25 * r * s**2
                + 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                - 0.25 * r**2 * s
                + 0.25 * r * s**2
                - 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                + 0.25 * r**2 * s
                + 0.25 * r * s**2
                + 0.25 * r * s
            ],
            [
                0.25 * r**2 * s**2
                + 0.25 * r**2 * s
                - 0.25 * r * s**2
                - 0.25 * r * s
            ],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 * s + 0.5 * s**2 - 0.5 * s],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 - 0.5 * r * s**2 + 0.5 * r],
            [-0.5 * r**2 * s**2 - 0.5 * r**2 * s + 0.5 * s**2 + 0.5 * s],
            [-0.5 * r**2 * s**2 + 0.5 * r**2 + 0.5 * r * s**2 - 0.5 * r],
            [1.0 * r**2 * s**2 - 1.0 * r**2 - 1.0 * s**2 + 1.0],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q9_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q9(pcoords[iP]))
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q9(pcoord: np.ndarray):
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_Q9(pcoord)
    res = np.zeros((3, 27), dtype=pcoord.dtype)
    for i in prange(9):
        res[:, i * 3 : (i + 1) * 3] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q9_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2, 18), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_Q9(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array(
        [
            [
                0.5 * r * s**2 - 0.5 * r * s - 0.25 * s**2 + 0.25 * s,
                0.5 * r**2 * s - 0.25 * r**2 - 0.5 * r * s + 0.25 * r,
            ],
            [
                0.5 * r * s**2 - 0.5 * r * s + 0.25 * s**2 - 0.25 * s,
                0.5 * r**2 * s - 0.25 * r**2 + 0.5 * r * s - 0.25 * r,
            ],
            [
                0.5 * r * s**2 + 0.5 * r * s + 0.25 * s**2 + 0.25 * s,
                0.5 * r**2 * s + 0.25 * r**2 + 0.5 * r * s + 0.25 * r,
            ],
            [
                0.5 * r * s**2 + 0.5 * r * s - 0.25 * s**2 - 0.25 * s,
                0.5 * r**2 * s + 0.25 * r**2 - 0.5 * r * s - 0.25 * r,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r * s,
                -1.0 * r**2 * s + 0.5 * r**2 + 1.0 * s - 0.5,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r - 0.5 * s**2 + 0.5,
                -1.0 * r**2 * s - 1.0 * r * s,
            ],
            [
                -1.0 * r * s**2 - 1.0 * r * s,
                -1.0 * r**2 * s - 0.5 * r**2 + 1.0 * s + 0.5,
            ],
            [
                -1.0 * r * s**2 + 1.0 * r + 0.5 * s**2 - 0.5,
                -1.0 * r**2 * s + 1.0 * r * s,
            ],
            [2.0 * r * s**2 - 2.0 * r, 2.0 * r**2 * s - 2.0 * s],
        ],
        dtype=pcoord.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q9_multi(pcoords: ndarray):
    """
    Returns the first orderderivatives of the shape functions,
    evaluated at multiple points, according to 'pcoords'.

    ---
    (nP, nNE, 2)
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q9(pcoords[iP])
    return res
