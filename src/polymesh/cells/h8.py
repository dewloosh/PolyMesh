# -*- coding: utf-8 -*-
from typing import Tuple, List
from sympy import symbols
import numpy as np
from numpy import ndarray

from neumann.numint import GaussPoints as Gauss

from ..polyhedron import HexaHedron
from ..utils.utils import cells_coords
from ..utils.cells.h8 import (shp_H8_multi, dshp_H8_multi, volumes_H8, 
                           shape_function_matrix_H8_multi)


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
    shpfnc = shp_H8_multi
    shpmfnc = shape_function_matrix_H8_multi
    dshpfnc = dshp_H8_multi
    
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
