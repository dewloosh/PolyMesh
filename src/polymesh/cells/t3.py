# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from ..polygon import Triangle

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_CST(pcoord: ndarray):
    r, s = pcoord[:2]
    return np.array([1, r, s], dtype=pcoord.dtype)


@njit(nogil=True, cache=__cache)
def shp_CST(pcoord: ndarray):
    r, s = pcoord
    return np.array([1 - r - s, r, s], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_CST_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_CST(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_CST(pcoord: np.ndarray):
    eye = np.eye(2, dtype=pcoord.dtype)
    shp = shp_CST(pcoord)
    res = np.zeros((2, 6), dtype=pcoord.dtype)
    for i in prange(3):
        res[:, i * 2: (i+1) * 2] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_CST(x):
    return np.array([[-1., -1.], [1., 0.], [0., 1.]])


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_CST_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_CST(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_CST_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_CST_bulk(pcoords[iP])
    return res


class T3(Triangle):
    """
    A class to handle 3-noded triangles.
    
    See Also
    --------
    :class:`polymesh.polygon.Triangle`
    
    """
    
    shpfnc = shp_CST_bulk
    dshpfnc = dshp_CST_bulk

    @classmethod
    def lcoords(cls, *args, **kwargs) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[0., 0.], [1., 0.], [0., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[1/3, 1/3]])
    
    @classmethod
    def shape_function_values(cls, coords: ndarray, *args, **kwargs) -> ndarray:
        """
        Evaluates the shape functions. The points of evaluation should be 
        understood on the master element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (3,) for a single, (N, 3) for N evaulation points.

        """
        return shp_CST_bulk(coords) if len(coords.shape) == 2 else shp_CST(coords)

    @classmethod
    def shape_function_derivatives(cls, coords: ndarray, *args, **kwargs) -> ndarray:
        """
        Returns shape function derivatives wrt. the master element. The points of 
        evaluation should be understood on the master element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (3, 2) for a single, (N, 3, 2) for N evaulation points.

        """
        return dshp_CST_bulk(coords) if len(coords.shape) == 2 else dshp_CST(coords)
