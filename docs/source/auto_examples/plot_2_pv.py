"""
Compound Meshes
===============

"""

from polymesh import PolyData
from polymesh.trimesh import TriMesh
from polymesh.grid import Grid
from polymesh.space import StandardFrame
import numpy as np

A = StandardFrame(dim=3)
tri = TriMesh(size=(100, 100), shape=(10, 10), frame=A)
grid2d = Grid(size=(100, 100), shape=(10, 10), eshape='Q4', frame=A)
grid3d = Grid(size=(100, 100, 100), shape=(8, 6, 2), eshape='H8', frame=A)

mesh = PolyData(frame=A)
mesh['tri', 'T3'] = tri.move(np.array([0., 0., -200]))
mesh['grids', 'Q4'] = grid2d.move(np.array([0., 0., 200]))
mesh['grids', 'H8'] = grid3d

mesh['tri', 'T3'].pointdata['values'] = np.full(tri.coords().shape[0], 5.)
mesh['grids', 'Q4'].pointdata['values'] = np.full(grid2d.coords().shape[0], 10.)
mesh['grids', 'H8'].pointdata['values'] = np.full(grid3d.coords().shape[0], -5.)

mesh.to_standard_form()
mesh.lock(create_mappers=True)

import pyvista as pv
from pyvista import themes

my_theme = themes.DarkTheme()
my_theme.color = 'red'
my_theme.lighting = False
my_theme.show_edges = True
my_theme.axes.box = True

pv.set_plot_theme(my_theme)

mesh.pvplot(off_screen=True, window_size = (600, 400), theme=my_theme,
            jupyter_backend='static')
