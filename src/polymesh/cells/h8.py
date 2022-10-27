# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np
from numpy import ndarray

from neumann.numint import GaussPoints as Gauss

from ..polyhedron import HexaHedron
from ..utils import cells_coords

__cache = True


@njit(nogil=True, cache=__cache)
def monoms_H8(pcoord: np.ndarray):
    r, s, t = pcoord
    return np.array([1, r, s, t, r*s, r*t, s*t, r*s*t])


@njit(nogil=True, cache=__cache)
def shp_H8(pcoord):
    r, s, t = pcoord
    return np.array([-0.125*r*s*t + 0.125*r*s + 0.125*r*t - 0.125*r +
                     0.125*s*t - 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t - 0.125*r*s - 0.125*r*t + 0.125*r +
                     0.125*s*t - 0.125*s - 0.125*t + 0.125,
                     -0.125*r*s*t + 0.125*r*s - 0.125*r*t + 0.125*r -
                     0.125*s*t + 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t - 0.125*r*s + 0.125*r*t - 0.125*r -
                     0.125*s*t + 0.125*s - 0.125*t + 0.125,
                     0.125*r*s*t + 0.125*r*s - 0.125*r*t - 0.125*r -
                     0.125*s*t - 0.125*s + 0.125*t + 0.125,
                     -0.125*r*s*t - 0.125*r*s + 0.125*r*t + 0.125*r -
                     0.125*s*t - 0.125*s + 0.125*t + 0.125,
                     0.125*r*s*t + 0.125*r*s + 0.125*r*t + 0.125*r +
                     0.125*s*t + 0.125*s + 0.125*t + 0.125,
                     -0.125*r*s*t - 0.125*r*s - 0.125*r*t - 0.125*r +
                     0.125*s*t + 0.125*s + 0.125*t + 0.125]
                    )


@njit(nogil=True, parallel=True, cache=__cache)
def shp_H8_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H8(pcoord: np.ndarray):
    eye = np.eye(3, dtype=pcoord.dtype)
    shp = shp_H8(pcoord)
    res = np.zeros((3, 24), dtype=pcoord.dtype)
    for i in prange(8):
        res[:, i*3: (i+1) * 3] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_H8(pcoord):
    r, s, t = pcoord
    return np.array(
        [[-0.125*s*t + 0.125*s + 0.125*t - 0.125,
         -0.125*r*t + 0.125*r + 0.125*t - 0.125,
         -0.125*r*s + 0.125*r + 0.125*s - 0.125],
         [0.125*s*t - 0.125*s - 0.125*t + 0.125,
         0.125*r*t - 0.125*r + 0.125*t - 0.125,
         0.125*r*s - 0.125*r + 0.125*s - 0.125],
         [-0.125*s*t + 0.125*s - 0.125*t + 0.125,
         -0.125*r*t + 0.125*r - 0.125*t + 0.125,
         -0.125*r*s - 0.125*r - 0.125*s - 0.125],
         [0.125*s*t - 0.125*s + 0.125*t - 0.125,
         0.125*r*t - 0.125*r - 0.125*t + 0.125,
         0.125*r*s + 0.125*r - 0.125*s - 0.125],
         [0.125*s*t + 0.125*s - 0.125*t - 0.125,
         0.125*r*t + 0.125*r - 0.125*t - 0.125,
         0.125*r*s - 0.125*r - 0.125*s + 0.125],
         [-0.125*s*t - 0.125*s + 0.125*t + 0.125,
         -0.125*r*t - 0.125*r - 0.125*t - 0.125,
         -0.125*r*s + 0.125*r - 0.125*s + 0.125],
         [0.125*s*t + 0.125*s + 0.125*t + 0.125,
         0.125*r*t + 0.125*r + 0.125*t + 0.125,
         0.125*r*s + 0.125*r + 0.125*s + 0.125],
         [-0.125*s*t - 0.125*s - 0.125*t - 0.125,
         -0.125*r*t - 0.125*r + 0.125*t + 0.125,
         -0.125*r*s - 0.125*r + 0.125*s + 0.125]]
    )


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_H8_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 8, 3), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = dshp_H8(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def volumes_H8(ecoords: np.ndarray, qpos: np.ndarray,
               qweight: np.ndarray):
    nE = ecoords.shape[0]
    volumes = np.zeros(nE, dtype=ecoords.dtype)
    nQ = len(qweight)
    for iQ in range(nQ):
        dshp = dshp_H8(qpos[iQ])
        for i in prange(nE):
            jac = ecoords[i].T @ dshp
            djac = np.linalg.det(jac)
            volumes[i] += qweight[iQ] * djac
    return volumes


class H8(HexaHedron):
    """
    8-node isoparametric hexahedron.

    top        
    7--6  
    |  |
    4--5

    bottom
    3--2  
    |  |
    0--1

    """

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[-1., -1., -1], [1., -1., -1.], [1., 1., -1.],
                         [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.],
                         [1., 1., 1.], [-1., 1., 1.]])

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
            An array of shape (8,) for a single, (N, 8) for N evaulation points.

        """
        if len(coords.shape) == 2:
            return dshp_H8_bulk(coords)
        else:
            return shp_H8(coords)

    @classmethod
    def shape_function_derivatives(cls, coords: ndarray, *args, **kwargs) -> ndarray:
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
            An array of shape (8, 3) for a single, (N, 8, 3) for N evaulation points.

        """
        if len(coords.shape) == 2:
            return dshp_H8_bulk(coords)
        else:
            return dshp_H8(coords)

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
        qpos, qweight = Gauss(2, 2, 2)
        return volumes_H8(ecoords, qpos, qweight)
