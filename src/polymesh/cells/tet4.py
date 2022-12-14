# -*- coding: utf-8 -*-
from typing import Tuple, List
from numba import njit, prange
import numpy as np
from numpy import ndarray
from sympy import symbols
from ..polyhedron import TetraHedron
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


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_TET4_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 12), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_TET4(pcoords[iP])
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
    shpmfnc = shape_function_matrix_TET4_multi
    dshpfnc = dshp_TET4_bulk
    
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
        locvars = r, s, t = symbols('r s t', real=True)
        monoms = [1, r, s, t, r*s, r*t, s*t, r**2, s**2, t**2]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        return np.array([
            [0., 0., 0.], 
            [1., 0., 0.], 
            [0., 1., 0.],
            [0., 0., 1.]])

    @classmethod
    def lcenter(cls) -> ndarray:
        return np.array([[1/3, 1/3, 1/3]])
    