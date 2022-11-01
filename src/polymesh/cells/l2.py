# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numba import njit, prange
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..line import Line

__cache = True


__all__ = ['L2']


@njit(nogil=True, cache=__cache)
def monoms2(r):
    """
    Evaluates the polynomial base at one location in the range [-1, 1].

    Parameters
    ----------
    r : float
        The point of evaluation.

    Returns
    -------
    numpy array of shape (2,)
    """
    return np.array([1, r])


@njit(nogil=True, cache=__cache)
def shp2(r):
    """
    Evaluates the shape functions at one location in the range [-1, 1]
    """
    return np.array([1 - r, 1 + r]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def shp2_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp2(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp2(r):
    return np.array([-1, 1]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def dshp2_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = dshp2(pcoords[iP])
    return res


class L2(Line):
    """
    2-Node line element.
    """
    
    shpfnc = shp2_bulk
    dshpfnc = dshp2_bulk

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
        locvars = r = symbols('r', real=True)
        monoms = [1, r]
        return [locvars], monoms
    
    @classmethod
    def lcoords(cls, *args, **kwargs) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([-1., 1.])

    @classmethod
    def lcenter(cls, *args, **kwargs) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0.])
    
    @classmethod
    def shape_function_values(cls, coords: ndarray, *args, **kwargs) -> ndarray:
        """
        Evaluates the shape functions. The points of evaluation should be 
        understood in the range [-1, 1].

        Parameters
        ----------
        coords : float or numpy.ndarray
            Points of evaluation. It should be scalar for a single point
            and a 1d array for several points. 

        Returns
        -------
        numpy.ndarray
            An array of shape (2,) for a single, (N, 2) for N evaulation points.

        """
        return shp2_bulk(coords) if isinstance(coords, ndarray) else shp2(coords)

    @classmethod
    def shape_function_derivatives(cls, coords: ndarray, *args, **kwargs) -> ndarray:
        """
        Returns shape function derivatives wrt. the master element. The points of 
        evaluation should be understood in the range [-1, 1].

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be scalar for a single point
            and a 1d array for several points.

        Returns
        -------
        numpy.ndarray
            An array of shape (2,) for a single, (N, 2) for N evaulation points.

        """
        return dshp2_bulk(coords) if isinstance(coords, ndarray) else dshp2(coords)
       