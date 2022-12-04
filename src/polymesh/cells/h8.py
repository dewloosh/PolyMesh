# -*- coding: utf-8 -*-
from typing import Tuple, List, Iterable
from sympy import symbols
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
def shp_H8_multi(pcoords: np.ndarray):
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


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_H8_multi(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 3, 24), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = shape_function_matrix_H8(pcoords[iP])
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
def dshp_H8_multi(pcoords: ndarray):
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

    ::
    
        top        
        7--6  
        |  |
        4--5

        bottom
        3--2  
        |  |
        0--1
        
    See Also
    --------
    :class:`HexaHedron`

    """
    
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
        monoms = [1, r, s, t, r*s, r*t, s*t, r*s*t]
        return locvars, monoms

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
    def shape_function_values(cls, coords: ndarray) -> ndarray:
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
            An array of shape (8,) for a single, (N, 8) for N 
            evaulation points.

        """
        coords = np.array(coords)
        if len(coords.shape) == 2:
            return shp_H8_multi(coords)  
        else:
            return shp_H8(coords)

    @classmethod
    def shape_function_derivatives(cls, coords: ndarray) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element. 

        Parameters
        ----------
        coords : numpy.ndarray
            Points of evaluation. It should be a 1d array for a single point
            and a 2d array for several points. In the latter case, the points
            should run along the first axis.

        Returns
        -------
        numpy.ndarray
            An array of shape (8, 3) for a single, (N, 8, 3) for N 
            evaulation points.

        """
        coords = np.array(coords)
        if len(coords.shape) == 2:
            return dshp_H8_multi(coords)  
        else:
            return dshp_H8(coords)

    @classmethod
    def shape_function_matrix(cls, pcoords:Iterable[float]) -> ndarray:
        """
        Evaluates the shape function matrix at one or multiple points.

        Parameters
        ----------
        pcoords : Iterable
            1d or 2d iterable of location point coordinates. For multuple
            points, the first axis goes along the points, the second along
            spatial dimensions.

        Returns
        -------
        numpy.ndarray
            The returned array has a shape of (nDOF, nDOF * nNE), where
            nNE and nDOF stand for the nodes per element and number of 
            degrees of freedom respectively. For multiple evaluation
            points, the shape is (nP, nDOF, nDOF * nNE).

        """
        pcoords = np.array(pcoords)
        if len(pcoords.shape) == 2:
            return shape_function_matrix_H8_multi(pcoords)    
        else:
            return shape_function_matrix_H8(pcoords)
    
    def volumes(self, coords:ndarray=None, topo:ndarray=None) -> ndarray:
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
