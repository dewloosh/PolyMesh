# -*- coding: utf-8 -*-
from enum import IntEnum, unique

_meshio_to_vtk = {"triangle": "VTK_TRIANGLE"}


@unique
class vtkCellTypes(IntEnum):
    # # Linear cells
    # VTK_EMPTY_CELL = 0
    # VTK_VERTEX = 1
    # VTK_POLY_VERTEX = 2
    VTK_LINE = 3
    # VTK_POLY_LINE = 4
    VTK_TRIANGLE = 5
    # VTK_TRIANGLE_STRIP = 6
    # VTK_POLYGON = 7
    # VTK_PIXEL = 8
    VTK_QUAD = 9
    VTK_TETRA = 10
    # VTK_VOXEL = 11
    VTK_HEXAHEDRON = 12
    VTK_WEDGE = 13
    # VTK_PYRAMID = 14
    # VTK_PENTAGONAL_PRISM = 15
    # VTK_HEXAGONAL_PRISM = 16

    # # Quadratic, isoparametric cells
    # VTK_QUADRATIC_EDGE = 21
    VTK_QUADRATIC_TRIANGLE = 22
    # VTK_QUADRATIC_QUAD = 23
    # VTK_QUADRATIC_POLYGON = 36
    VTK_QUADRATIC_TETRA = 24
    # VTK_QUADRATIC_HEXAHEDRON = 25
    # VTK_QUADRATIC_WEDGE = 26
    # VTK_QUADRATIC_PYRAMID = 27
    VTK_BIQUADRATIC_QUAD = 28
    VTK_TRIQUADRATIC_HEXAHEDRON = 29
    # VTK_QUADRATIC_LINEAR_QUAD = 30
    # VTK_QUADRATIC_LINEAR_WEDGE = 31
    VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32
    # VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    # VTK_BIQUADRATIC_TRIANGLE = 34

    # # Cubic, isoparametric cell
    # VTK_CUBIC_LINE = 35

    # # Special class of cells formed by convex group of points
    # VTK_CONVEX_POINT_SET = 41

    # # Polyhedron cell (consisting of polygonal faces)
    # VTK_POLYHEDRON = 42

    # # Higher order cells in parametric form
    # VTK_PARAMETRIC_CURVE = 51
    # VTK_PARAMETRIC_SURFACE = 52
    # VTK_PARAMETRIC_TRI_SURFACE = 53
    # VTK_PARAMETRIC_QUAD_SURFACE = 54
    # VTK_PARAMETRIC_TETRA_REGION = 55
    # VTK_PARAMETRIC_HEX_REGION = 56

    # # Higher order cells
    # VTK_HIGHER_ORDER_EDGE = 60
    # VTK_HIGHER_ORDER_TRIANGLE = 61
    # VTK_HIGHER_ORDER_QUAD = 62
    # VTK_HIGHER_ORDER_POLYGON = 63
    # VTK_HIGHER_ORDER_TETRAHEDRON = 64
    # VTK_HIGHER_ORDER_WEDGE = 65
    # VTK_HIGHER_ORDER_PYRAMID = 66
    # VTK_HIGHER_ORDER_HEXAHEDRON = 67

    # # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    # VTK_LAGRANGE_CURVE = 68
    # VTK_LAGRANGE_TRIANGLE = 69
    # VTK_LAGRANGE_QUADRILATERAL = 70
    # VTK_LAGRANGE_TETRAHEDRON = 71
    # VTK_LAGRANGE_HEXAHEDRON = 72
    # VTK_LAGRANGE_WEDGE = 73
    # VTK_LAGRANGE_PYRAMID = 74

    # # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    # VTK_BEZIER_CURVE = 75
    # VTK_BEZIER_TRIANGLE = 76
    # VTK_BEZIER_QUADRILATERAL = 77
    # VTK_BEZIER_TETRAHEDRON = 78
    # VTK_BEZIER_HEXAHEDRON = 79
    # VTK_BEZIER_WEDGE = 80
    # VTK_BEZIER_PYRAMID = 81

    @staticmethod
    def celltype(ID):
        if isinstance(ID, int):
            celltype = celltype_by_value(ID)
        elif isinstance(ID, str):
            celltype = celltype_by_name(ID)
        return celltype.name, celltype.value


def celltype_by_name(name: str):
    return vtkCellTypes[name]


def celltype_by_value(value: int):
    return vtkCellTypes(value)


def meshio_to_vtk(category: str, default=None):
    try:
        return _meshio_to_vtk[category]
    except:
        return default


def CellTypeId(celltype):
    try:
        return vtkCellTypes[celltype].value
    except:
        return None
