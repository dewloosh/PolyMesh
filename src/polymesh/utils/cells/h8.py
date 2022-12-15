from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True


@njit(nogil=True, cache=__cache)
def shp_H8(pcoord: ndarray) -> ndarray:
    r, s, t = pcoord
    return np.array([-0.125*r*s*t + 0.125*r*s + 0.125*r*t - 0.125*r +
                     0.125*s*t - 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t - 0.125*r*s - 0.125*r*t + 0.125*r +
                     0.125*s*t - 0.125*s - 0.125*t + 0.125,
                     -0.125*r*s*t + 0.125*r*s - 0.125*r*t + 0.125*r -
                     0.125*s*t + 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t - 0.125*r*s + 0.125*r*t - 0.125*r -
                     0.125*s*t + 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t + 0.125*r*s - 0.125*r*t - 0.125*r -
                     0.125*s*t - 0.125*s + 0.125*t + 0.125,
                     -0.125*r*s*t - 0.125*r*s + 0.125*r*t + 0.125*r -
                     0.125*s*t - 0.125*s + 0.125*t + 0.125,
                     0.125*r*s*t + 0.125*r*s + 0.125*r*t + 0.125*r +
                     0.125*s*t + 0.125*s + 0.125*t + 0.125,
                     -0.125*r*s*t - 0.125*r*s - 0.125*r*t - 0.125*r +
                     0.125*s*t + 0.125*s + 0.125*t + 0.125]
                    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_H8_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H8(pcoord: ndarray) -> ndarray:
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_H8(pcoord)
    res = np.zeros((3, 24), dtype=pcoord.dtype)
    for i in prange(8):
        res[:, i*3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H8_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 24), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_H8(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp_H8(pcoord: ndarray) -> ndarray:
    r, s, t = pcoord
    return np.array(
        [[-0.125*s*t + 0.125*s + 0.125*t - 0.125,
         -0.125*r*t + 0.125*r + 0.125*t - 0.125,
         -0.125*r*s + 0.125*r + 0.125*s - 0.125],
         [0.125*s*t - 0.125*s - 0.125*t + 0.125,
         0.125*r*t - 0.125*r + 0.125*t - 0.125,
         0.125*r*s - 0.125*r + 0.125*s - 0.125],
         [-0.125*s*t + 0.125*s - 0.125*t + 0.125,
         -0.125*r*t + 0.125*r - 0.125*t + 0.125,
         -0.125*r*s - 0.125*r - 0.125*s - 0.125],
         [0.125*s*t - 0.125*s + 0.125*t - 0.125,
         0.125*r*t - 0.125*r - 0.125*t + 0.125,
         0.125*r*s + 0.125*r - 0.125*s - 0.125],
         [0.125*s*t + 0.125*s - 0.125*t - 0.125,
         0.125*r*t + 0.125*r - 0.125*t - 0.125,
         0.125*r*s - 0.125*r - 0.125*s + 0.125],
         [-0.125*s*t - 0.125*s + 0.125*t + 0.125,
         -0.125*r*t - 0.125*r - 0.125*t - 0.125,
         -0.125*r*s + 0.125*r - 0.125*s + 0.125],
         [0.125*s*t + 0.125*s + 0.125*t + 0.125,
         0.125*r*t + 0.125*r + 0.125*t + 0.125,
         0.125*r*s + 0.125*r + 0.125*s + 0.125],
         [-0.125*s*t - 0.125*s - 0.125*t - 0.125,
         -0.125*r*t - 0.125*r + 0.125*t + 0.125,
         -0.125*r*s - 0.125*r + 0.125*s + 0.125]]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_H8_multi(pcoords: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes_H8(ecoords: ndarray, qpos: ndarray,
               qweight: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshp_H8(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes