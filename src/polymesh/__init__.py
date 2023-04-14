__version__ = "0.0.19"
__description__ = "A Python package to build, manipulate and analyze polygonal meshes."

from .space import PointCloud
from .space import CartesianFrame
from .polydata import PolyData
from .linedata import LineData
from .linedata import LineData as PolyData1d
from .pointdata import PointData
from .utils import k_nearest_neighbours as KNN
from .topoarray import TopologyArray
from .trimesh import TriMesh
from .tetmesh import TetMesh
from .triang import triangulate
from .grid import grid, Grid
from .tetrahedralize import tetrahedralize
