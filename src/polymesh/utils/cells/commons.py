from typing import Callable
from numba import njit, prange
import numpy as np
from numpy import ndarray


__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_multi(pcoords: np.ndarray, shpfnc:Callable,
                                nDOF:int=3, nNE:int=8) -> ndarray:
    nP = pcoords.shape[0]
    eye = np.eye(nDOF, dtype=pcoords.dtype)
    res = np.zeros((nP, nDOF, nDOF*nNE), dtype=pcoords.dtype)
    for iP in prange(nP):
        shp = shpfnc(pcoords[iP])
        for i in prange(nNE):
            res[iP, :, i*nDOF: (i+1) * nDOF] = eye*shp[i]
    return res
