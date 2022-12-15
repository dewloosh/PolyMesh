from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def shp_T3(pcoord: ndarray):
    r, s = pcoord
    return np.array([1 - r - s, r, s], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_T3_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_T3(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_T3(pcoord: np.ndarray):
    eye = np.eye(2, dtype=pcoord.dtype)
    shp = shp_T3(pcoord)
    res = np.zeros((3, 9), dtype=pcoord.dtype)
    for i in prange(3):
        res[:, i * 3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_T3_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 9), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_T3(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp_T3(x):
    return np.array([[-1., -1.], [1., 0.], [0., 1.]])


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_T3_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_T3(pcoords[iP])
    return res