# -*- coding: utf-8 -*-
from ...mesh.utils import cells_coords
from ...mesh.polygon import QuadraticTriangle as Triangle
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


@njit(nogil=True, cache=__cache)
def monoms_LST(pcoord: ndarray):
    r, s = pcoord[0:2]
    return np.array([1, r, s, r * s, r * r, s * s], 
                    dtype=pcoord.dtype)


@njit(nogil=True, cache=__cache)
def shp_LST(pcoord: np.ndarray):
    r, s = pcoord[0:2]
    return np.array([2.0*r**2 + 4.0*r*s - 3.0*r +
                     2.0*s**2 - 3.0*s + 1.0,
                     2.0*r**2 - 1.0*r,
                     2.0*s**2 - 1.0*s,
                     -4.0*r**2 - 4.0*r*s + 4.0*r,
                     4.0*r*s,
                     -4.0*r*s - 4.0*s**2 + 4.0*s],
                    dtype=pcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_LST(pcoord: np.ndarray):
    eye = np.eye(2, dtype=pcoord.dtype)
    shp = shp_LST(pcoord)
    res = np.zeros((2, 12), dtype=pcoord.dtype)
    for i in prange(6):
        res[:, i * 2: (i+1) * 2] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_LST(pcoord):
    r, s = pcoord[0:2]
    return np.array([[4.0*r + 4.0*s - 3.0, 4.0*r + 4.0*s - 3.0],
                     [4.0*r - 1.0, 0],
                     [0, 4.0*s - 1.0],
                     [-8.0*r - 4.0*s + 4.0, -4.0*r],
                     [4.0*s, 4.0*r],
                     [-4.0*s, -4.0*r - 8.0*s + 4.0]])


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_LST_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 6, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_LST(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def areas_T6(ecoords: np.ndarray, qpos: np.ndarray, qweight: np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    nP = len(qweight)
    for i in range(nP):
        dshp = dshp_LST(qpos[i])
        for iE in prange(nE):
            jac = ecoords[iE].T @ dshp
            djac = np.linalg.det(jac)
            res[iE] += qweight[i] * djac
    return res


class T6(Triangle):
    """
    A class to handle 6-noded triangles.
    
    """

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                         [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([[1/3, 1/3]])

    def shape_function_derivatives(self, coords=None, *args, **kwargs):
        coords = self.pointdata.x if coords is None else coords
        if len(coords.shape) == 2:
            return dshp_LST_bulk(coords)
        else:
            return dshp_LST(coords)

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        ecoords = cells_coords(coords[:, :2], topo)
        qpos, qweight = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]), \
            np.array([1/6, 1/6, 1/6])
        return areas_T6(ecoords, qpos, qweight)
    