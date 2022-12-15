from numba import njit, prange
import numpy as np
from numpy import ndarray
from neumann import flatten2dC

__cache = True


@njit(nogil=True, cache=__cache)
def shp_Q4(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([
        [0.25*(1-r)*(1-s)],
        [0.25*(1+r)*(1-s)],
        [0.25*(1+r)*(1+s)],
        [0.25*(1-r)*(1+s)]
    ], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q4_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q4(pcoords[iP]))
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q4(pcoord: ndarray):
    r, s = pcoord[:2]
    return np.array([[(s - 1)/4, (r - 1)/4],
                     [(1 - s)/4, (-r - 1)/4],
                     [(s + 1)/4, (r + 1)/4],
                     [(-s - 1)/4, (1 - r)/4]],
                    dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q4_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q4(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q4(pcoord: ndarray) -> ndarray:
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_Q4(pcoord)
    res = np.zeros((3, 12), dtype=pcoord.dtype)
    for i in prange(4):
        res[:, i*3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q4_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 12), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_Q4(pcoords[iP])
    return res