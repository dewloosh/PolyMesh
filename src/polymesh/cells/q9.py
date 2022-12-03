# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from neumann.array import flatten2dC

from ..polygon import BiQuadraticQuadrilateral

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_powers_Q9():
    return np.array([
        [0, 0], [1, 0], [0, 1],
        [1, 1], [2, 0], [0, 2],
        [2, 1], [1, 2], [2, 2]])


@njit(nogil=True, cache=__cache)
def monoms_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([1, r, s, r*s, r**2, s**2, r**2*s,
                     r*s**2, r**2*s**2],
                    dtype=pcoord.dtype)


@njit(nogil=True, cache=__cache)
def shp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([
        [0.25*r**2*s**2 - 0.25*r**2*s - 0.25*r*s**2 + 0.25*r*s],
        [0.25*r**2*s**2 - 0.25*r**2*s + 0.25*r*s**2 - 0.25*r*s],
        [0.25*r**2*s**2 + 0.25*r**2*s + 0.25*r*s**2 + 0.25*r*s],
        [0.25*r**2*s**2 + 0.25*r**2*s - 0.25*r*s**2 - 0.25*r*s],
        [-0.5*r**2*s**2 + 0.5*r**2*s + 0.5*s**2 - 0.5*s],
        [-0.5*r**2*s**2 + 0.5*r**2 - 0.5*r*s**2 + 0.5*r],
        [-0.5*r**2*s**2 - 0.5*r**2*s + 0.5*s**2 + 0.5*s],
        [-0.5*r**2*s**2 + 0.5*r**2 + 0.5*r*s**2 - 0.5*r],
        [1.0*r**2*s**2 - 1.0*r**2 - 1.0*s**2 + 1.0]
    ], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q9_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q9(pcoords[iP]))
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_Q9(pcoord: np.ndarray):
    eye = np.eye(2, dtype=pcoord.dtype)
    shp = shp_Q9(pcoord)
    res = np.zeros((2, 18), dtype=pcoord.dtype)
    for i in prange(9):
        res[:, i*2: (i+1) * 2] = eye * shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q9(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([
        [0.5*r*s**2 - 0.5*r*s - 0.25*s**2 + 0.25*s,
         0.5*r**2*s - 0.25*r**2 - 0.5*r*s + 0.25*r],
        [0.5*r*s**2 - 0.5*r*s + 0.25*s**2 - 0.25*s,
         0.5*r**2*s - 0.25*r**2 + 0.5*r*s - 0.25*r],
        [0.5*r*s**2 + 0.5*r*s + 0.25*s**2 + 0.25*s,
         0.5*r**2*s + 0.25*r**2 + 0.5*r*s + 0.25*r],
        [0.5*r*s**2 + 0.5*r*s - 0.25*s**2 - 0.25*s,
         0.5*r**2*s + 0.25*r**2 - 0.5*r*s - 0.25*r],
        [-1.0*r*s**2 + 1.0*r*s,
         -1.0*r**2*s + 0.5*r**2 + 1.0*s - 0.5],
        [-1.0*r*s**2 + 1.0*r - 0.5*s**2 + 0.5,
         -1.0*r**2*s - 1.0*r*s],
        [-1.0*r*s**2 - 1.0*r*s,
         -1.0*r**2*s - 0.5*r**2 + 1.0*s + 0.5],
        [-1.0*r*s**2 + 1.0*r + 0.5*s**2 - 0.5,
         -1.0*r**2*s + 1.0*r*s],
        [2.0*r*s**2 - 2.0*r,
         2.0*r**2*s - 2.0*s]
    ], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q9_bulk(pcoords: ndarray):
    """
    Returns the first orderderivatives of the shape functions, 
    evaluated at multiple points, according to 'pcoords'.
    
    ---
    (nP, nNE, 2)
    """
    nP = pcoords.shape[0]
    res = np.zeros((nP, 9, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q9(pcoords[iP])
    return res 


class Q9(BiQuadraticQuadrilateral):
    """
    Polygon class for 9-noded biquadratic quadrilaterals. 

    See Also
    --------
    :class:`BiQuadraticQuadrilateral`
    
    """
    
    shpfnc = shp_Q9_bulk
    dshpfnc = dshp_Q9_bulk

    @classmethod
    def lcoords(cls, *args, **kwargs):
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.],
                        [0., -1.], [1., 0.], [0., 1.], [-1., 0.], [0., 0.]])

    @classmethod
    def lcenter(cls, *args, **kwargs) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0., 0.])

    @classmethod
    def shape_function_values(cls, pcoords: ndarray, 
                              *args, **kwargs) -> ndarray:
        """
        Evaluates the shape functions. The points of evaluation should be 
        understood in the range [-1, 1].

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (9,) for a single, (N, 9) for N evaulation points.

        """
        return shp_Q9_bulk(pcoords) if len(pcoords.shape) == 2 else shp_Q9(pcoords)

    @classmethod
    def shape_function_derivatives(cls, pcoords: ndarray, 
                                   *args, **kwargs) -> ndarray:
        """
        Returns shape function derivatives wrt. the master element. The points of 
        evaluation should be understood in the range [-1, 1].

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (9, 2) for a single, (N, 9, 2) for N evaulation points.

        """
        return dshp_Q9_bulk(pcoords) if len(pcoords.shape) == 2 else dshp_Q9(pcoords)
