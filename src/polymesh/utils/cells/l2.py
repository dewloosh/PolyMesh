from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def shp_L2(r):
    """
    Evaluates the shape functions at one location in the range [-1, 1].
    """
    return np.array([1 - r, 1 + r]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def shp_L2_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_L2(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_L2(pcoord: ndarray) -> ndarray:
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_L2(pcoord)
    res = np.zeros((3, 6), dtype=pcoord.dtype)
    for i in prange(2):
        res[:, i*3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_L2_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 6), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_L2(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp_L2(r):
    return np.array([-1, 1]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_L2_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = dshp_L2(pcoords[iP])
    return res