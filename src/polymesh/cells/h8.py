from typing import Union, Iterable, Tuple, List

from sympy import symbols
import numpy as np
from numpy import ndarray

from neumann.numint import gauss_points as gp
from neumann import atleast2d

from ..polyhedron import HexaHedron
from ..utils.utils import cells_coords
from ..utils.cells.h8 import (
    shp_H8_multi,
    dshp_H8_multi,
    volumes_H8,
    shape_function_matrix_H8_multi,
    monoms_H8,
    _pip_H8_bulk_,
    _pip_H8_bulk_knn_
)
from ..utils.cells.gauss import Gauss_Legendre_Hex_Grid
from ..utils.knn import k_nearest_neighbours

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
    :class:`~polymesh.polyhedron.HexaHedron`
    """

    shpfnc = shp_H8_multi
    shpmfnc = shape_function_matrix_H8_multi
    dshpfnc = dshp_H8_multi
    monomsfnc = monoms_H8

    quadrature = {
        "full": Gauss_Legendre_Hex_Grid(2, 2, 2),
    }

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
        locvars = r, s, t = symbols("r s t", real=True)
        monoms = [1, r, s, t, r * s, r * t, s * t, r * s * t]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array(
            [
                [-1.0, -1.0, -1],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0, 0.0, 0.0])

    def volumes(self) -> ndarray:
        """
        Returns the volumes of the cells.

        Returns
        -------
        numpy.ndarray
        """
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = gp(2, 2, 2)
        return volumes_H8(ecoords, qpos, qweight)
    
    def pip(
        self, x: Union[Iterable, ndarray], 
        tol: float = 1e-12,
        lazy:bool=True, 
        k:int=4, 
        tetrahedralize: bool=True,
    ) -> Union[bool, ndarray]:
        """
        Returns an 1d boolean integer array that tells if the points specified by 'x'
        are included in any of the cells in the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.
        lazy: bool, Optional
            If False, the ckeck is performed for all cells in the block. If True,
            it is used in combination with parameter 'k' and the check is only performed
            for the k nearest neighbours of the input points. Default is True.
        k: int, Optional
            The number of neighbours for the case when 'lazy' is true. Default is 4.
        tetrahedralize: bool, Optional
            Wether to perform the check on a tetrahedralized version of the block or not.

        Returns
        -------
        bool or numpy.ndarray
            A single or NumPy array of booleans for every input point.
        """
        if tetrahedralize:
            return super().pip(x, tol, lazy, k)
        else:
            x = atleast2d(x, front=True)
            x_loc = self.glob_to_loc(x)
            pips = _pip_H8_bulk_(x_loc, tol)
            
            if lazy:
                centers = self.centers()
                k = min(4, len(centers))
                knn = k_nearest_neighbours(centers, x, k=k)
                pips = _pip_H8_bulk_knn_(x_loc, knn, tol)
            else:
                pips = _pip_H8_bulk_(x_loc, tol)
            
            return np.squeeze(np.any(pips, axis=1))
