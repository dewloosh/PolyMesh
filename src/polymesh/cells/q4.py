# -*- coding: utf-8 -*-
from ...math.array import flatten2dC
from numba import njit, prange
import numpy as np
from numpy import ndarray

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

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., -1.], [1., -1.],
                        [1., 1.], [-1., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0., 0.])

    def shape_function_values(self, pcoords, *args, **kwargs):
        if len(pcoords.shape) == 2:
            return shp_Q4_bulk(pcoords)
        else:
            return shp_Q4(pcoords)

    def shape_function_derivatives(self, pcoords, *args, **kwargs):
        if len(pcoords.shape) == 2:
            return dshp_Q4_bulk(pcoords)
        else:
            return dshp_Q4(pcoords)
