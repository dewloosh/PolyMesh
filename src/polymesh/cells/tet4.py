# -*- coding: utf-8 -*-
from ..polyhedron import TetraHedron
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
def shp_TET4_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_TET4(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET4(pcoord: np.ndarray):
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_TET4(pcoord)
    res = np.zeros((3, 12), dtype=pcoord.dtype)
    for i in prange(4):
        res[:, i * 3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_TET4(x):
    return np.array([[-1., -1., -1.], [1., 0., 0.], 
                     [0., 1., 0.], [0., 0., 1.]])
    

@njit(nogil=True, parallel=True, cache=__cache)
def dshp_TET4_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 4, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_TET4(pcoords[iP])
    return res


class TET4(TetraHedron):
    """
    4-node isoparametric hexahedron.
    
    See Also
    --------
    :class:`TetraHedron`
    
    """
    
    shpfnc = shp_TET4_bulk
    dshpfnc = dshp_TET4_bulk

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
        """
        Returns shape function derivatives wrt. the master element. The points of 
        evaluation should be understood in the master element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (4, 3) for a single, (N, 4, 3) for N evaulation points.

        """
        return dshp_TET4_bulk(coords) if len(coords.shape) == 2 else dshp_TET4(coords)
    