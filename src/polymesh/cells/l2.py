# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..line import Line
from ..utils.cells.l2 import shp_L2_multi, dshp_L2_multi, shape_function_matrix_L2_multi


__all__ = ["L2"]


class L2(Line):
    """
    2-Node line element.

    See Also
    --------
    :class:`Line`
    """

    shpfnc = shp_L2_multi
    shpmfnc = shape_function_matrix_L2_multi
    dshpfnc = dshp_L2_multi

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
        locvars = r = symbols("r", real=True)
        monoms = [1, r]
        return [locvars], monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([-1.0, 1.0])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray

        """
        return np.array([0.0])
