# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def tet_vol_bulk(ecoords: ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        v1 = ecoords[i, 1] - ecoords[i, 0]
        v2 = ecoords[i, 2] - ecoords[i, 0]
        v3 = ecoords[i, 3] - ecoords[i, 0]
        res[i] = np.dot(np.cross(v1, v2), v3)
    return np.abs(res) / 6


if __name__ == '__main__':
    pass