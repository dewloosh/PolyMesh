# -*- coding: utf-8 -*-
import numpy as np

from .polygon import Triangle
from .cell import PolyCell3d
from .utils.topology import H8_to_TET4


class PolyHedron(PolyCell3d):
    """Base class to handle polyhedra."""

    _face_cls_ = Triangle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TetraHedron(PolyHedron):
    """Class for 4-noded tetrahedra."""

    NNODE = 4
    vtkCellType = 10
    __label__ = "TET4"

    def to_tetrahedra(self) -> np.ndarray:
        return self.topology().to_numpy()


class QuadraticTetraHedron(PolyHedron):
    """Class for 10-noded quadratic tetrahedra."""

    NNODE = 10
    vtkCellType = 24
    __label__ = "TET10"


class HexaHedron(PolyHedron):
    """Class for 8-noded hexahedra."""

    NNODE = 8
    vtkCellType = 12
    __label__ = "H8"

    def to_tetrahedra(self) -> np.ndarray:
        return H8_to_TET4(None, self.topology().to_numpy())[1]


class TriquadraticHexaHedron(PolyHedron):
    """Class for 27-noded triquadratic hexahedra."""

    NNODE = 27
    vtkCellType = 29
    __label__ = "H27"


class Wedge(PolyHedron):
    """Class for 6-noded trilinear wedges."""

    NNODE = 6
    vtkCellType = 13
    __label__ = "W6"


class BiquadraticWedge(PolyHedron):
    """Class for 6-noded biquadratic wedges."""

    NNODE = 18
    vtkCellType = 32
    __label__ = "W18"
