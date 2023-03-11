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
        2D array of shape (nNE, nD) containing coordinates global for every node of
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
    """
    Global to local transformation for a single point and cell.

    Returns local coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of global
    coordinates and a function to evaluate the monomials of the shape
    functions.

    Parameters
    ----------
    gcoords: numpy.ndarray
        3D array of shape (nE, nNE, nD) containing global coordinates for every
        node of several elements.
    x: numpy.ndarray
        2D array of global coordinates of shape (nP, nD) of several points.
    lcoords: numpy.ndarray
        2D array of local coordinates of the parametric element.
    monomsfnc: Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape (nP, nE, nD) of parametric cooridnates of the specified
        points.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nD number of components.
    """
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
