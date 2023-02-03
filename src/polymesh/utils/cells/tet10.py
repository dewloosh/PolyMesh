from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def shp_TET10(pcoord: ndarray):
    r, s, t = pcoord
    u = 1 - r - s - t
    return np.array(
        [
            u * (2 * u - 1),
            r * (2 * r - 1),
            s * (2 * s - 1),
            t * (2 * t - 1),
            4 * u * r,
            4 * r * s,
            4 * s * u,
            4 * u * t,
            4 * r * t,
            4 * s * t,
        ]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_TET10_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_TET10(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET10(pcoord: np.ndarray, ndof:int=3):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_TET10(pcoord)
    res = np.zeros((ndof, ndof*10), dtype=pcoord.dtype)
    for i in prange(10):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET10_multi(pcoords: np.ndarray, ndof:int=3):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof*10), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_TET10(pcoords[iP], ndof)
    return res


@njit(nogil=True, cache=__cache)
def dshp_TET10(x):
    return np.array(
        [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_TET10_multi(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 10, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_TET10(pcoords[iP])
    return res
