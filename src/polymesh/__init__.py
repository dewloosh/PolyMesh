# -*- coding: utf-8 -*-
__version__ = "0.0.1a"
__description__ = "A Python package to build, manipulate and analyze polygonal meshes."

from .space import PointCloud
from .space import CartesianFrame
from .tri.triang import triangulate
from .grid import grid, Grid
from .tri.trimesh import TriMesh
from .tet.tetmesh import TetMesh
from .polydata import PolyData
from .linedata import LineData
from .linedata import LineData as PolyData1d
from .pointdata import PointData