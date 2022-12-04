# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..line import QuadraticLine


__all__ = ['L3']


class L3(QuadraticLine):
    """
    3-Node line element.
    
    See Also
    --------
    :class:`QuadraticLine`
    """
    
    shpfnc = None
    dshpfnc = None
    
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
        locvars = r = symbols('r', real=True)
        monoms = [1, r, r**2]
        return [locvars], monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([-1., 0., 1.])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0.])

    """def shape_function_values(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return shp3_bulk(coords)
        else:
            return shp3(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp3_bulk(coords)
        else:
            return dshp3(coords)"""
