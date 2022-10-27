# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from neumann.numint import GaussPoints as Gauss

from ..polyhedron import TriquadraticHexaHedron
from ..utils import cells_coords

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_H27(pcoord: np.ndarray):
    r, s, t = pcoord
    return np.array([1, r, s, t, s*t, r*t, r*s, r*s*t, r**2, s**2, t**2,
                     r**2*s, r*s**2, r*t**2, r**2*t, s**2*t, s*t**2, r**2*s*t,
                     r*s**2*t, r*s*t**2, r**2*s**2, s**2*t**2, r**2*t**2,
                     r**2*s**2*t**2, r**2*s**2*t, r**2*s*t**2, r*s**2*t**2])


@njit(nogil=True, cache=__cache)
def shp_H27(pcoord):
    r, s, t = pcoord
    return np.array(
        [0.125*r**2*s**2*t**2 - 0.125*r**2*s**2*t - 0.125*r**2*s*t**2 +
         0.125*r**2*s*t - 0.125*r*s**2*t**2 + 0.125*r*s**2*t +
         0.125*r*s*t**2 - 0.125*r*s*t,
         0.125*r**2*s**2*t**2 - 0.125*r**2*s**2*t - 0.125*r**2*s*t**2 +
         0.125*r**2*s*t + 0.125*r*s**2*t**2 - 0.125*r*s**2*t -
         0.125*r*s*t**2 + 0.125*r*s*t,
         0.125*r**2*s**2*t**2 - 0.125*r**2*s**2*t + 0.125*r**2*s*t**2 -
         0.125*r**2*s*t + 0.125*r*s**2*t**2 - 0.125*r*s**2*t +
         0.125*r*s*t**2 - 0.125*r*s*t,
         0.125*r**2*s**2*t**2 - 0.125*r**2*s**2*t + 0.125*r**2*s*t**2 -
         0.125*r**2*s*t - 0.125*r*s**2*t**2 + 0.125*r*s**2*t -
         0.125*r*s*t**2 + 0.125*r*s*t,
         0.125*r**2*s**2*t**2 + 0.125*r**2*s**2*t - 0.125*r**2*s*t**2 -
         0.125*r**2*s*t - 0.125*r*s**2*t**2 - 0.125*r*s**2*t +
         0.125*r*s*t**2 + 0.125*r*s*t,
         0.125*r**2*s**2*t**2 + 0.125*r**2*s**2*t - 0.125*r**2*s*t**2 -
         0.125*r**2*s*t + 0.125*r*s**2*t**2 + 0.125*r*s**2*t -
         0.125*r*s*t**2 - 0.125*r*s*t,
         0.125*r**2*s**2*t**2 + 0.125*r**2*s**2*t + 0.125*r**2*s*t**2 +
         0.125*r**2*s*t + 0.125*r*s**2*t**2 + 0.125*r*s**2*t +
         0.125*r*s*t**2 + 0.125*r*s*t,
         0.125*r**2*s**2*t**2 + 0.125*r**2*s**2*t + 0.125*r**2*s*t**2 +
         0.125*r**2*s*t - 0.125*r*s**2*t**2 - 0.125*r*s**2*t -
         0.125*r*s*t**2 - 0.125*r*s*t,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2*t + 0.25*r**2*s*t**2 -
         0.25*r**2*s*t + 0.25*s**2*t**2 - 0.25*s**2*t - 0.25*s*t**2 + 0.25*s*t,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2*t + 0.25*r**2*t**2 -
         0.25*r**2*t - 0.25*r*s**2*t**2 + 0.25*r*s**2*t +
         0.25*r*t**2 - 0.25*r*t,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2*t - 0.25*r**2*s*t**2 +
         0.25*r**2*s*t + 0.25*s**2*t**2 - 0.25*s**2*t +
         0.25*s*t**2 - 0.25*s*t,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2*t + 0.25*r**2*t**2 -
         0.25*r**2*t + 0.25*r*s**2*t**2 - 0.25*r*s**2*t -
         0.25*r*t**2 + 0.25*r*t,
         -0.25*r**2*s**2*t**2 - 0.25*r**2*s**2*t + 0.25*r**2*s*t**2 +
         0.25*r**2*s*t + 0.25*s**2*t**2 + 0.25*s**2*t - 0.25*s*t**2 - 0.25*s*t,
         -0.25*r**2*s**2*t**2 - 0.25*r**2*s**2*t + 0.25*r**2*t**2 +
         0.25*r**2*t - 0.25*r*s**2*t**2 - 0.25*r*s**2*t +
         0.25*r*t**2 + 0.25*r*t,
         -0.25*r**2*s**2*t**2 - 0.25*r**2*s**2*t - 0.25*r**2*s*t**2 -
         0.25*r**2*s*t + 0.25*s**2*t**2 + 0.25*s**2*t + 0.25*s*t**2 + 0.25*s*t,
         -0.25*r**2*s**2*t**2 - 0.25*r**2*s**2*t + 0.25*r**2*t**2 +
         0.25*r**2*t + 0.25*r*s**2*t**2 + 0.25*r*s**2*t -
         0.25*r*t**2 - 0.25*r*t,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2 + 0.25*r**2*s*t**2 -
         0.25*r**2*s + 0.25*r*s**2*t**2 - 0.25*r*s**2 -
         0.25*r*s*t**2 + 0.25*r*s,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2 + 0.25*r**2*s*t**2 -
         0.25*r**2*s - 0.25*r*s**2*t**2 + 0.25*r*s**2 +
         0.25*r*s*t**2 - 0.25*r*s,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2 - 0.25*r**2*s*t**2 +
         0.25*r**2*s - 0.25*r*s**2*t**2 + 0.25*r*s**2 -
         0.25*r*s*t**2 + 0.25*r*s,
         -0.25*r**2*s**2*t**2 + 0.25*r**2*s**2 - 0.25*r**2*s*t**2 +
         0.25*r**2*s + 0.25*r*s**2*t**2 - 0.25*r*s**2 +
         0.25*r*s*t**2 - 0.25*r*s,
         0.5*r**2*s**2*t**2 - 0.5*r**2*s**2 - 0.5*r**2*t**2 + 0.5*r**2 -
         0.5*r*s**2*t**2 + 0.5*r*s**2 + 0.5*r*t**2 - 0.5*r,
         0.5*r**2*s**2*t**2 - 0.5*r**2*s**2 - 0.5*r**2*t**2 + 0.5*r**2 +
         0.5*r*s**2*t**2 - 0.5*r*s**2 - 0.5*r*t**2 + 0.5*r,
         0.5*r**2*s**2*t**2 - 0.5*r**2*s**2 - 0.5*r**2*s*t**2 + 0.5*r**2*s -
         0.5*s**2*t**2 + 0.5*s**2 + 0.5*s*t**2 - 0.5*s,
         0.5*r**2*s**2*t**2 - 0.5*r**2*s**2 + 0.5*r**2*s*t**2 - 0.5*r**2*s -
         0.5*s**2*t**2 + 0.5*s**2 - 0.5*s*t**2 + 0.5*s,
         0.5*r**2*s**2*t**2 - 0.5*r**2*s**2*t - 0.5*r**2*t**2 + 0.5*r**2*t -
         0.5*s**2*t**2 + 0.5*s**2*t + 0.5*t**2 - 0.5*t,
         0.5*r**2*s**2*t**2 + 0.5*r**2*s**2*t - 0.5*r**2*t**2 - 0.5*r**2*t -
         0.5*s**2*t**2 - 0.5*s**2*t + 0.5*t**2 + 0.5*t,
         -1.0*r**2*s**2*t**2 + 1.0*r**2*s**2 + 1.0*r**2*t**2 - 1.0*r**2 +
         1.0*s**2*t**2 - 1.0*s**2 - 1.0*t**2 + 1.0])


@njit(nogil=True, parallel=True, cache=__cache)
def shp_H27_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 27), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_H27(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H27(pcoord: np.ndarray):
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_H27(pcoord)
    res = np.zeros((3, 24), dtype=pcoord.dtype)
    for i in prange(8):
        res[:, i*3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_H27(pcoord):
    r, s, t = pcoord
    return np.array([
        [0.25*r*s**2*t**2 - 0.25*r*s**2*t - 0.25*r*s*t**2 + 0.25*r*s*t -
            0.125*s**2*t**2 + 0.125*s**2*t + 0.125*s*t**2 - 0.125*s*t,
            0.25*r**2*s*t**2 - 0.25*r**2*s*t - 0.125*r**2*t**2 + 0.125*r**2*t
            - 0.25*r*s*t**2 + 0.25*r*s*t + 0.125*r*t**2 - 0.125*r*t,
            0.25*r**2*s**2*t - 0.125*r**2*s**2 - 0.25*r**2*s*t + 0.125*r**2*s
            - 0.25*r*s**2*t + 0.125*r*s**2 + 0.25*r*s*t - 0.125*r*s],
        [0.25*r*s**2*t**2 - 0.25*r*s**2*t - 0.25*r*s*t**2 + 0.25*r*s*t +
            0.125*s**2*t**2 - 0.125*s**2*t - 0.125*s*t**2 + 0.125*s*t,
            0.25*r**2*s*t**2 - 0.25*r**2*s*t - 0.125*r**2*t**2 + 0.125*r**2*t
            + 0.25*r*s*t**2 - 0.25*r*s*t - 0.125*r*t**2 + 0.125*r*t,
            0.25*r**2*s**2*t - 0.125*r**2*s**2 - 0.25*r**2*s*t + 0.125*r**2*s
            + 0.25*r*s**2*t - 0.125*r*s**2 - 0.25*r*s*t + 0.125*r*s],
        [0.25*r*s**2*t**2 - 0.25*r*s**2*t + 0.25*r*s*t**2 - 0.25*r*s*t +
            0.125*s**2*t**2 - 0.125*s**2*t + 0.125*s*t**2 - 0.125*s*t,
            0.25*r**2*s*t**2 - 0.25*r**2*s*t + 0.125*r**2*t**2 - 0.125*r**2*t
            + 0.25*r*s*t**2 - 0.25*r*s*t + 0.125*r*t**2 - 0.125*r*t,
            0.25*r**2*s**2*t - 0.125*r**2*s**2 + 0.25*r**2*s*t - 0.125*r**2*s
            + 0.25*r*s**2*t - 0.125*r*s**2 + 0.25*r*s*t - 0.125*r*s],
        [0.25*r*s**2*t**2 - 0.25*r*s**2*t + 0.25*r*s*t**2 - 0.25*r*s*t -
            0.125*s**2*t**2 + 0.125*s**2*t - 0.125*s*t**2 + 0.125*s*t,
            0.25*r**2*s*t**2 - 0.25*r**2*s*t + 0.125*r**2*t**2 - 0.125*r**2*t
            - 0.25*r*s*t**2 + 0.25*r*s*t - 0.125*r*t**2 + 0.125*r*t,
            0.25*r**2*s**2*t - 0.125*r**2*s**2 + 0.25*r**2*s*t - 0.125*r**2*s
            - 0.25*r*s**2*t + 0.125*r*s**2 - 0.25*r*s*t + 0.125*r*s],
        [0.25*r*s**2*t**2 + 0.25*r*s**2*t - 0.25*r*s*t**2 - 0.25*r*s*t -
            0.125*s**2*t**2 - 0.125*s**2*t + 0.125*s*t**2 + 0.125*s*t,
            0.25*r**2*s*t**2 + 0.25*r**2*s*t - 0.125*r**2*t**2 - 0.125*r**2*t
            - 0.25*r*s*t**2 - 0.25*r*s*t + 0.125*r*t**2 + 0.125*r*t,
            0.25*r**2*s**2*t + 0.125*r**2*s**2 - 0.25*r**2*s*t - 0.125*r**2*s
            - 0.25*r*s**2*t - 0.125*r*s**2 + 0.25*r*s*t + 0.125*r*s],
        [0.25*r*s**2*t**2 + 0.25*r*s**2*t - 0.25*r*s*t**2 - 0.25*r*s*t +
            0.125*s**2*t**2 + 0.125*s**2*t - 0.125*s*t**2 - 0.125*s*t,
            0.25*r**2*s*t**2 + 0.25*r**2*s*t - 0.125*r**2*t**2 - 0.125*r**2*t
            + 0.25*r*s*t**2 + 0.25*r*s*t - 0.125*r*t**2 - 0.125*r*t,
            0.25*r**2*s**2*t + 0.125*r**2*s**2 - 0.25*r**2*s*t - 0.125*r**2*s
            + 0.25*r*s**2*t + 0.125*r*s**2 - 0.25*r*s*t - 0.125*r*s],
        [0.25*r*s**2*t**2 + 0.25*r*s**2*t + 0.25*r*s*t**2 + 0.25*r*s*t +
            0.125*s**2*t**2 + 0.125*s**2*t + 0.125*s*t**2 + 0.125*s*t,
            0.25*r**2*s*t**2 + 0.25*r**2*s*t + 0.125*r**2*t**2 + 0.125*r**2*t
            + 0.25*r*s*t**2 + 0.25*r*s*t + 0.125*r*t**2 + 0.125*r*t,
            0.25*r**2*s**2*t + 0.125*r**2*s**2 + 0.25*r**2*s*t + 0.125*r**2*s
            + 0.25*r*s**2*t + 0.125*r*s**2 + 0.25*r*s*t + 0.125*r*s],
        [0.25*r*s**2*t**2 + 0.25*r*s**2*t + 0.25*r*s*t**2 + 0.25*r*s*t -
            0.125*s**2*t**2 - 0.125*s**2*t - 0.125*s*t**2 - 0.125*s*t,
            0.25*r**2*s*t**2 + 0.25*r**2*s*t + 0.125*r**2*t**2 + 0.125*r**2*t
            - 0.25*r*s*t**2 - 0.25*r*s*t - 0.125*r*t**2 - 0.125*r*t,
            0.25*r**2*s**2*t + 0.125*r**2*s**2 + 0.25*r**2*s*t + 0.125*r**2*s
            - 0.25*r*s**2*t - 0.125*r*s**2 - 0.25*r*s*t - 0.125*r*s],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2*t + 0.5*r*s*t**2 - 0.5*r*s*t,
            -0.5*r**2*s*t**2 + 0.5*r**2*s*t + 0.25*r**2*t**2 - 0.25*r**2*t +
            0.5*s*t**2 - 0.5*s*t - 0.25*t**2 + 0.25*t,
            -0.5*r**2*s**2*t + 0.25*r**2*s**2 + 0.5*r**2*s*t - 0.25*r**2*s +
            0.5*s**2*t - 0.25*s**2 - 0.5*s*t + 0.25*s],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2*t + 0.5*r*t**2 - 0.5*r*t -
            0.25*s**2*t**2 + 0.25*s**2*t + 0.25*t**2 - 0.25*t,
            -0.5*r**2*s*t**2 + 0.5*r**2*s*t - 0.5*r*s*t**2 + 0.5*r*s*t,
            -0.5*r**2*s**2*t + 0.25*r**2*s**2 + 0.5*r**2*t - 0.25*r**2 -
            0.5*r*s**2*t + 0.25*r*s**2 + 0.5*r*t - 0.25*r],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2*t - 0.5*r*s*t**2 + 0.5*r*s*t,
            -0.5*r**2*s*t**2 + 0.5*r**2*s*t - 0.25*r**2*t**2 + 0.25*r**2*t +
            0.5*s*t**2 - 0.5*s*t + 0.25*t**2 - 0.25*t,
            -0.5*r**2*s**2*t + 0.25*r**2*s**2 - 0.5*r**2*s*t + 0.25*r**2*s +
            0.5*s**2*t - 0.25*s**2 + 0.5*s*t - 0.25*s],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2*t + 0.5*r*t**2 - 0.5*r*t +
            0.25*s**2*t**2 - 0.25*s**2*t - 0.25*t**2 + 0.25*t,
            -0.5*r**2*s*t**2 + 0.5*r**2*s*t + 0.5*r*s*t**2 - 0.5*r*s*t,
            -0.5*r**2*s**2*t + 0.25*r**2*s**2 + 0.5*r**2*t - 0.25*r**2 +
            0.5*r*s**2*t - 0.25*r*s**2 - 0.5*r*t + 0.25*r],
        [-0.5*r*s**2*t**2 - 0.5*r*s**2*t + 0.5*r*s*t**2 + 0.5*r*s*t,
            -0.5*r**2*s*t**2 - 0.5*r**2*s*t + 0.25*r**2*t**2 + 0.25*r**2*t +
            0.5*s*t**2 + 0.5*s*t - 0.25*t**2 - 0.25*t,
            -0.5*r**2*s**2*t - 0.25*r**2*s**2 + 0.5*r**2*s*t + 0.25*r**2*s +
            0.5*s**2*t + 0.25*s**2 - 0.5*s*t - 0.25*s],
        [-0.5*r*s**2*t**2 - 0.5*r*s**2*t + 0.5*r*t**2 + 0.5*r*t -
            0.25*s**2*t**2 - 0.25*s**2*t + 0.25*t**2 + 0.25*t,
            -0.5*r**2*s*t**2 - 0.5*r**2*s*t - 0.5*r*s*t**2 - 0.5*r*s*t,
            -0.5*r**2*s**2*t - 0.25*r**2*s**2 + 0.5*r**2*t + 0.25*r**2 -
            0.5*r*s**2*t - 0.25*r*s**2 + 0.5*r*t + 0.25*r],
        [-0.5*r*s**2*t**2 - 0.5*r*s**2*t - 0.5*r*s*t**2 - 0.5*r*s*t,
            -0.5*r**2*s*t**2 - 0.5*r**2*s*t - 0.25*r**2*t**2 - 0.25*r**2*t +
            0.5*s*t**2 + 0.5*s*t + 0.25*t**2 + 0.25*t,
            -0.5*r**2*s**2*t - 0.25*r**2*s**2 - 0.5*r**2*s*t - 0.25*r**2*s +
            0.5*s**2*t + 0.25*s**2 + 0.5*s*t + 0.25*s],
        [-0.5*r*s**2*t**2 - 0.5*r*s**2*t + 0.5*r*t**2 + 0.5*r*t +
            0.25*s**2*t**2 + 0.25*s**2*t - 0.25*t**2 - 0.25*t,
            -0.5*r**2*s*t**2 - 0.5*r**2*s*t + 0.5*r*s*t**2 + 0.5*r*s*t,
            -0.5*r**2*s**2*t - 0.25*r**2*s**2 + 0.5*r**2*t + 0.25*r**2 +
            0.5*r*s**2*t + 0.25*r*s**2 - 0.5*r*t - 0.25*r],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2 + 0.5*r*s*t**2 - 0.5*r*s +
            0.25*s**2*t**2 - 0.25*s**2 - 0.25*s*t**2 + 0.25*s,
            -0.5*r**2*s*t**2 + 0.5*r**2*s + 0.25*r**2*t**2 - 0.25*r**2 +
            0.5*r*s*t**2 - 0.5*r*s - 0.25*r*t**2 + 0.25*r,
            -0.5*r**2*s**2*t + 0.5*r**2*s*t + 0.5*r*s**2*t - 0.5*r*s*t],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2 + 0.5*r*s*t**2 - 0.5*r*s -
            0.25*s**2*t**2 + 0.25*s**2 + 0.25*s*t**2 - 0.25*s,
            -0.5*r**2*s*t**2 + 0.5*r**2*s + 0.25*r**2*t**2 - 0.25*r**2 -
            0.5*r*s*t**2 + 0.5*r*s + 0.25*r*t**2 - 0.25*r,
            -0.5*r**2*s**2*t + 0.5*r**2*s*t - 0.5*r*s**2*t + 0.5*r*s*t],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2 - 0.5*r*s*t**2 + 0.5*r*s -
            0.25*s**2*t**2 + 0.25*s**2 - 0.25*s*t**2 + 0.25*s,
            -0.5*r**2*s*t**2 + 0.5*r**2*s - 0.25*r**2*t**2 + 0.25*r**2 -
            0.5*r*s*t**2 + 0.5*r*s - 0.25*r*t**2 + 0.25*r,
            -0.5*r**2*s**2*t - 0.5*r**2*s*t - 0.5*r*s**2*t - 0.5*r*s*t],
        [-0.5*r*s**2*t**2 + 0.5*r*s**2 - 0.5*r*s*t**2 + 0.5*r*s +
            0.25*s**2*t**2 - 0.25*s**2 + 0.25*s*t**2 - 0.25*s,
            -0.5*r**2*s*t**2 + 0.5*r**2*s - 0.25*r**2*t**2 + 0.25*r**2 +
            0.5*r*s*t**2 - 0.5*r*s + 0.25*r*t**2 - 0.25*r,
            -0.5*r**2*s**2*t - 0.5*r**2*s*t + 0.5*r*s**2*t + 0.5*r*s*t],
        [1.0*r*s**2*t**2 - 1.0*r*s**2 - 1.0*r*t**2 + 1.0*r -
            0.5*s**2*t**2 + 0.5*s**2 + 0.5*t**2 - 0.5,
            1.0*r**2*s*t**2 - 1.0*r**2*s - 1.0*r*s*t**2 + 1.0*r*s,
            1.0*r**2*s**2*t - 1.0*r**2*t - 1.0*r*s**2*t + 1.0*r*t],
        [1.0*r*s**2*t**2 - 1.0*r*s**2 - 1.0*r*t**2 + 1.0*r +
            0.5*s**2*t**2 - 0.5*s**2 - 0.5*t**2 + 0.5,
            1.0*r**2*s*t**2 - 1.0*r**2*s + 1.0*r*s*t**2 - 1.0*r*s,
            1.0*r**2*s**2*t - 1.0*r**2*t + 1.0*r*s**2*t - 1.0*r*t],
        [1.0*r*s**2*t**2 - 1.0*r*s**2 - 1.0*r*s*t**2 + 1.0*r*s,
            1.0*r**2*s*t**2 - 1.0*r**2*s - 0.5*r**2*t**2 + 0.5*r**2 -
            1.0*s*t**2 + 1.0*s + 0.5*t**2 - 0.5,
            1.0*r**2*s**2*t - 1.0*r**2*s*t - 1.0*s**2*t + 1.0*s*t],
        [1.0*r*s**2*t**2 - 1.0*r*s**2 + 1.0*r*s*t**2 - 1.0*r*s,
            1.0*r**2*s*t**2 - 1.0*r**2*s + 0.5*r**2*t**2 - 0.5*r**2 -
            1.0*s*t**2 + 1.0*s - 0.5*t**2 + 0.5,
            1.0*r**2*s**2*t + 1.0*r**2*s*t - 1.0*s**2*t - 1.0*s*t],
        [1.0*r*s**2*t**2 - 1.0*r*s**2*t - 1.0*r*t**2 + 1.0*r*t,
            1.0*r**2*s*t**2 - 1.0*r**2*s*t - 1.0*s*t**2 + 1.0*s*t,
            1.0*r**2*s**2*t - 0.5*r**2*s**2 - 1.0*r**2*t + 0.5*r**2 -
            1.0*s**2*t + 0.5*s**2 + 1.0*t - 0.5],
        [1.0*r*s**2*t**2 + 1.0*r*s**2*t - 1.0*r*t**2 - 1.0*r*t,
            1.0*r**2*s*t**2 + 1.0*r**2*s*t - 1.0*s*t**2 - 1.0*s*t,
            1.0*r**2*s**2*t + 0.5*r**2*s**2 - 1.0*r**2*t - 0.5*r**2 -
            1.0*s**2*t - 0.5*s**2 + 1.0*t + 0.5],
        [-2.0*r*s**2*t**2 + 2.0*r*s**2 + 2.0*r*t**2 - 2.0*r,
            -2.0*r**2*s*t**2 + 2.0*r**2*s + 2.0*s*t**2 - 2.0*s,
            -2.0*r**2*s**2*t + 2.0*r**2*t + 2.0*s**2*t - 2.0*t]])


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_H27_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 27, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_H27(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes_H27(ecoords: np.ndarray, qpos: np.ndarray,
                qweight: np.ndarray):
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshp_H27(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes


class H27(TriquadraticHexaHedron):
    """
    27-node isoparametric triquadratic hexahedron
    
    top
    7---14---6
    |    |   |
    15--25--13
    |    |   |
    4---12---5

    middle
    19--23--18
    |    |   |
    20--26--21
    |    |   |
    16--22--17

    bottom
    3---10---2
    |    |   |
    11--24---9
    |    |   |
    0----8---1

    """

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([
            [-1., -1., -1], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.],
            [-1., -1., 1.], [1., -1., 1.],  [1., 1., 1.],  [-1., 1., 1.],
            [0., -1., -1.], [1., 0., -1.],  [0., 1., -1.], [-1., 0., -1.],
            [0., -1., 1.],  [1., 0., 1.],   [0., 1., 1.],  [-1., 0., 1.],
            [-1., -1., 0.], [1., -1., 0.],  [1., 1., 0.],  [-1., 1., 0.],
            [-1., 0., 0.],  [1., 0., 0.],   [0., -1., 0.], [0., 1., 0.],
            [0., 0., -1.],  [0., 0., 1.],   [0., 0., 0.]])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0., 0., 0.])
    
    @classmethod
    def shape_function_values(cls, coords: ndarray, *args, **kwargs) -> ndarray:
        """
        Evaluates the shape functions. The points of evaluation should be 
        understood in the master element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (27,) for a single, (N, 27) for N evaulation points.

        """
        if len(coords.shape) == 2:
            return dshp_H27_bulk(coords)
        else:
            return shp_H27(coords)

    @classmethod
    def shape_function_derivatives(cls, coords=None, *args, **kwargs) -> ndarray:
        """
        Returns shape function derivatives wrt. the master element. The points of evaluation 
        should be understood in the master element.

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (27, 3) for a single, (N, 27, 3) for N evaulation points.

        """
        if len(coords.shape) == 2:
            return dshp_H27_bulk(coords)
        else:
            return dshp_H27(coords)

    def volumes(self, coords=None, topo=None) -> ndarray:
        """
        Returns the volumes of the cells.

        Returns
        -------
        numpy.ndarray

        """
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        ecoords = cells_coords(coords, topo)
        qpos, qweight = Gauss(3, 3, 3)
        return volumes_H27(ecoords, qpos, qweight)
