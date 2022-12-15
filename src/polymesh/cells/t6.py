# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..utils.utils import cells_coords
from ..polygon import QuadraticTriangle as Triangle
from ..utils.cells.t6 import (shp_T6_multi, dshp_T6_multi, areas_T6,
                              shape_function_matrix_T6_multi)


class T6(Triangle):
    """
    A class to handle 6-noded triangles.
    
    See Also
    --------
    :class:`Triangle`
    """
    shpfnc = shp_T6_multi
    shpmfnc = shape_function_matrix_T6_multi
    dshpfnc = dshp_T6_multi
    
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
        locvars = r, s = symbols('r s', real=True)
        monoms = [1, r, s, r**2, s**2, r*s]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                         [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([[1/3, 1/3]])
    
    def areas(self) -> ndarray:
        """
        Returns the areas of the triangles of the block.
        
        Returns
        -------
        numpy.ndarray
        
        """
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords[:, :2], topo)
        qpos, qweight = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]), \
            np.array([1/6, 1/6, 1/6])
        return areas_T6(ecoords, qpos, qweight)
    