from typing import Callable
from numba import njit, prange
import numpy as np
from numpy import ndarray

from neumann.linalg import inv

__cache = True


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes(ecoords: ndarray, dshp: ndarray, qweight: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        _dshp = dshp[iQ]
        for i in prange(nE):
            jac = ecoords[i].T @ _dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes


@njit(nogil=True, parallel=True, cache=__cache)
def _loc_to_glob_bulk_(shp: ndarray, gcoords: ndarray) -> ndarray:
    """
    Local to global transformation for several cells and points.

    Parameters
    ----------
    shp : numpy.ndarray
        The shape functions evaluated at a 'nP' number of local coordinates.
    gcoords : numpy.ndarray
        3D array of shape (nE, nNE, nD) containing coordinates global for every node of
        a single element.

    Returns
    -------
    numpy.ndarray
        Array of global cooridnates of shape (nE, nP, nD).
    """
    nP = shp.shape[-1]
    nE, _, nD = gcoords.shape
    res = np.zeros((nE, nP, nD), shp.dtype)
    for i in prange(nE):
        res[i, :, :] = shp @ gcoords[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_loc_bulk_(
    lcoords: ndarray, monoms_glob_cells: ndarray, monoms_glob_points: ndarray
) -> ndarray:
    nP = monoms_glob_points.shape[0]
    nE = monoms_glob_cells.shape[0]
    nD = lcoords.shape[-1]
    res = np.zeros((nE, nP, nD), dtype=lcoords.dtype)
    for i in prange(nE):
        coeffs = inv(monoms_glob_cells[i]).T
        for k in prange(nP):
            shp = coeffs @ monoms_glob_points[k]
            res[i, k, :] = shp @ lcoords
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_loc_bulk_2_(
    lcoords: ndarray, monoms_glob_cells: ndarray, monoms_glob_points: ndarray
) -> ndarray:
    nP = monoms_glob_points.shape[0]
    nD = lcoords.shape[-1]
    res = np.zeros((nP, nD), dtype=lcoords.dtype)
    for i in prange(nP):
        coeffs = inv(monoms_glob_cells[i]).T
        shp = coeffs @ monoms_glob_points[i]
        res[i, :] = shp @ lcoords
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _ntet_to_loc_bulk_(
    lcoords: ndarray,
    nat_tet: ndarray,
    tetmap: ndarray,
    cell_tet_indices: ndarray,
) -> ndarray:
    nP = cell_tet_indices.shape[0]
    res = np.zeros((nP, 3), dtype=lcoords.dtype)
    for i in prange(nP):
        i_tet_rel = cell_tet_indices[i]
        nat = nat_tet[i, i_tet_rel]
        subtopo = tetmap[i_tet_rel]
        for j in range(subtopo.shape[0]):
            res[i] += lcoords[subtopo[j]] * nat[j]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _find_first_hits_(pips: ndarray) -> ndarray:
    nP, nE = pips.shape
    res = np.zeros(nP, dtype=np.int64)
    for i in prange(nP):
        for j in prange(nE):
            if pips[i, j]:
                res[i] = j
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _find_first_hits_knn_(pips: ndarray, neighbours: ndarray) -> ndarray:
    nP, nE = pips.shape
    res = np.zeros(nP, dtype=np.int64)
    for i in prange(nP):
        for j in prange(nE):
            if pips[i, j]:
                res[i] = neighbours[i, j]
    return res
