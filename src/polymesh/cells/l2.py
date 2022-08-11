# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange
import numpy as np
from numpy import ndarray

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

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0.])
    
    def shape_function_values(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return shp2_bulk(coords)
        else:
            return shp2(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp2_bulk(coords)
        else:
            return dshp2(coords)
       