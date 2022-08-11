# -*- coding: utf-8 -*-
from ...mesh.polyhedron import TetraHedron
from ...math.array import repeat
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


@njit(nogil=True, cache=__cache)
def monoms_TET4(pcoord: ndarray):
    r, s, t = pcoord
    return np.array([1, r, s, t])


@njit(nogil=True, cache=__cache)
def shp_TET4(pcoord: ndarray):
    r, s, t = pcoord
    return np.array([1 - r - s - t, r, s, t])


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET4(pcoord: np.ndarray):
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_TET4(pcoord)
    res = np.zeros((3, 12), dtype=pcoord.dtype)
    for i in prange(4):
        res[:, i * 3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_TET4():
    return np.array([[-1., -1., -1.], [1., 0., 0.], 
                     [0., 1., 0.], [0., 0., 1.]])


class TET4(TetraHedron):

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([
            [0., 0., 0.], 
            [1., 0., 0.], 
            [0., 1., 0.],
            [0., 0., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([[1/3, 1/3, 1/3]])

    def shape_function_derivatives(self, coords=None, *args, **kwargs):
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        if len(coords.shape) == 2:
            return repeat(dshp_TET4(), coords.shape[0])
        else:
            return dshp_TET4()