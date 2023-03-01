from typing import Callable
from numba import njit, prange
import numpy as np
from numpy import ndarray

from neumann.linalg import inv

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_multi(
    pcoords: np.ndarray, shpfnc: Callable, nDOF: int = 3, nNE: int = 8
) -> ndarray:
    nP = pcoords.shape[0]
    eye = np.eye(nDOF, dtype=pcoords.dtype)
    res = np.zeros((nP, nDOF, nDOF * nNE), dtype=pcoords.dtype)
    for iP in prange(nP):
        shp = shpfnc(pcoords[iP])
        for i in prange(nNE):
            res[iP, :, i * nDOF : (i + 1) * nDOF] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes(
    ecoords: ndarray, qpos: ndarray, qweight: ndarray, dshpfnc: Callable
) -> ndarray:
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshpfnc(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes2(ecoords: ndarray, dshp: ndarray, qweight: ndarray) -> ndarray:
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


@njit(nogil=True, cache=__cache)
def loc_to_glob(shp: ndarray, gcoords: ndarray) -> ndarray:
    """
    Local to global transformation for a single cell and point.

    Returns global coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of parametric
    coordinates and a function to evaluate the shape functions.

    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space
    shp : (nNE, nP) ndarray
        The shape functions evaluated at a nP number of local coordinates.
    
    Returns
    -------
    (nD, ) ndarray
        Global cooridnates of the specified point.
    """
    return gcoords.T @ shp


@njit(nogil=True, parallel=True, cache=__cache)
def glob_to_loc(
    coord: np.ndarray, gcoords: np.ndarray, lcoords: np.ndarray, monomsfnc: Callable
):
    """
    Global to local transformation for a single point and cell.

    Returns local coordinates of a point in an element, provided the global
    corodinates of the points of the element, an array of global
    coordinates and a function to evaluate the monomials of the shape
    functions.

    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space
    coord : (nD, ) ndarray
        1D array of global coordinates for a single point.
            nD : number of dimensions of the model space
    lcoords : (nNE, nDP) ndarray
        2D array of local coordinates of the parametric element.
            nNE : number of vertices of the element
            nDP : number of dimensions of the parametric space
    monomsfnc : Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.

    Returns
    -------
    (nDP, ) ndarray
        Parametric cooridnates of the specified point.

    Notes
    -----
    'shpfnc' must be a numba-jitted function, that accepts a 1D array of
    exactly nDP number of components.
    """
    nNE = gcoords.shape[0]
    monoms = np.zeros((nNE, nNE), dtype=coord.dtype)
    for i in prange(nNE):
        monoms[:, i] = monomsfnc(gcoords[i])
    coeffs = inv(monoms)
    shp = coeffs @ monomsfnc(coord)
    return lcoords.T @ shp


@njit(nogil=True, cache=__cache)
def point_in_polygon(
    coord: np.ndarray,
    gcoords: np.ndarray,
    lcoords: np.ndarray,
    monomsfnc: Callable,
    shpfnc: Callable,
    tol=1e-12,
):
    """
    Point-in-polygon test for a single cell and point.

    Performs the point-in-poligon test for a single point and cell.
    False means that the point is outside of the cell, True means the point
    is either inside, or on the boundary of the cell.
    
    ::note
        This function is Numba-jittable in 'nopython' mode.
    
    Parameters
    ----------
    gcoords : (nNE, nD) ndarray
        2D array containing coordinates for every node of a single element.
            nNE : number of vertices of the element
            nD : number of dimensions of the model space
    coord : (nD, ) ndarray
        1D array of global coordinates for a single point.
            nD : number of dimensions of the model space
    lcoords : (nNE, nDP) ndarray
        2D array of local coordinates of the parametric element.
            nNE : number of vertices of the element
            nDP : number of dimensions of the parametric space
    monomsfnc : Callable
        A function that evaluates monomials of the shape functions at a point
        specified with parametric coordinates.
    shpfnc : Callable
        A function that evaluates shape function values at a point,
        specified with parametric coordinates.

    Returns
    -------
    bool
        False if points is outside of the cell, True otherwise.

    Notes
    -----
    'shpfnc'  and 'monomsfnc' must be numba-jitted functions, that accept
    a 1D array of exactly nDP number of components, where nDP is the number
    of paramatric cooridnate dimensions.
    """
    limit = 1 + tol
    loc = glob_to_loc(coord, gcoords, lcoords, monomsfnc)
    return np.all(shpfnc(loc) <= limit)