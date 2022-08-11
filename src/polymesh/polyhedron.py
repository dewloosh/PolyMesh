# -*- coding: utf-8 -*-
import numpy as np

from .polygon import Triangle
from .cell import PolyCell2d, PolyCell3d
from .topo.tr import H8_to_TET4


class PolyHedron(PolyCell3d):
    
    _face_cls_ = Triangle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TetraHedron(PolyHedron):

    NNODE = 4
    vtkCellType = 10
    __label__ = 'TET4'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self) -> np.ndarray:
        return self.topology().to_numpy()


class QuadraticTetraHedron(PolyHedron):

    NNODE = 10
    vtkCellType = 24
    __label__ = 'TET10'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HexaHedron(PolyHedron):

    NNODE = 8
    vtkCellType = 12
    __label__ = 'H8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def to_tetrahedra(self) -> np.ndarray:
        return H8_to_TET4(None, self.topology().to_numpy())[1]


class TriquadraticHexaHedron(PolyHedron):

    NNODE = 27
    vtkCellType = 29
    __label__ = 'H27'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wedge(PolyHedron):

    NNODE = 6
    vtkCellType = 13
    __label__ = 'W6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiquadraticWedge(PolyHedron):

    NNODE = 18
    vtkCellType = 32
    __label__ = 'W18'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)