# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from sympy import symbols

from ..polyhedron import QuadraticTetraHedron
from ..utils.cells.tet10 import (shp_TET10_multi, dshp_TET10_multi,
                                 shape_function_matrix_TET10_multi)


class TET10(QuadraticTetraHedron):
    """
    10-node isoparametric hexahedron.

    See Also
    --------
    :class:`QuadraticTetraHedron`
    """
    """shpfnc = shp_TET10_multi
    shpmfnc = shape_function_matrix_TET10_multi
    dshpfnc = dshp_TET10_multi"""

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
    def lcoords(cls):
        return np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [.5, 0., 0.],
            [.5, .5, 0.],
            [0., .5, 0.],
            [0., 0., .5],
            [.5, .0, .5],
            [0., .5, .5],
        ])

    @classmethod
    def lcenter(cls):
        return np.array([[1/3, 1/3, 1/3]])
