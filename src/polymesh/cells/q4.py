# -*- coding: utf-8 -*-
from typing import Tuple, List
from neumann.array import flatten2dC
from numba import njit, prange
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..polygon import Quadrilateral

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_Q4(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([1, r, s, r*s], dtype=pcoord.dtype)


@njit(nogil=True, cache=__cache)
def shp_Q4(pcoord: np.ndarray):
    r, s = pcoord[:2]
    return np.array([
        [0.25*(1-r)*(1-s)],
        [0.25*(1+r)*(1-s)],
        [0.25*(1+r)*(1+s)],
        [0.25*(1-r)*(1+s)]
    ], dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shp_Q4_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = flatten2dC(shp_Q4(pcoords[iP]))
    return res


@njit(nogil=True, cache=__cache)
def dshp_Q4(pcoord: ndarray):
    r, s = pcoord[:2]
    return np.array([[(s - 1)/4, (r - 1)/4],
                     [(1 - s)/4, (-r - 1)/4],
                     [(s + 1)/4, (r + 1)/4],
                     [(-s - 1)/4, (1 - r)/4]],
                    dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_Q4_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_Q4(pcoords[iP])
    return res


class Q4(Quadrilateral):
    """
    Polygon class for 4-noded bilinear quadrilaterals. 

    See Also
    --------
    :class:`Quadrilateral`

    """
    
    shpfnc = shp_Q4_bulk
    dshpfnc = dshp_Q4_bulk

    @classmethod
    def polybase(cls) -> Tuple[List]:
        """
        Retruns the polynomial base of the master element.

        Returns
        -------
        list
            A list of SymPy symbols.
        list
            A list of monomials.

        """
        locvars = r, s = symbols('r, s', real=True)
        monoms = [1, r, s, r*s]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[-1., -1.], [1., -1.],
                        [1., 1.], [-1., 1.]])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0., 0.])

    @classmethod
    def shape_function_values(cls, pcoords: ndarray) -> ndarray:
        """
        Evaluates the shape functions. The points of evaluation 
        should be understood on the master element

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a 
            single point and a 2d array for several points. In the 
            latter case, the points should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (4,) for a single, (N, 4) for N evaulation 
            points.

        """
        pcoords = np.array(pcoords)
        if len(pcoords.shape) == 2:  
            return shp_Q4_bulk(pcoords)
        else: 
            return shp_Q4(pcoords)

    @classmethod
    def shape_function_derivatives(cls, pcoords: ndarray) -> ndarray:
        """
        Returns shape function derivatives wrt. the master element. 
        The points of evaluation should be understood on the master 
        element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single 
            point and a 2d array for several points. In the latter case, 
            the points should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (4, 2) for a single, (N, 4, 2) for N 
            evaulation points.

        """
        pcoords = np.array(pcoords)
        if len(pcoords.shape) == 2:  
            return dshp_Q4_bulk(pcoords)
        else: 
            return dshp_Q4(pcoords)