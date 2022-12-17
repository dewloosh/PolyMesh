"""
A Simple Cube
=============

"""

from polymesh import PolyData, PointData
from polymesh.grid import grid
from polymesh.space import StandardFrame
from polymesh.cells import H27

size = Lx, Ly, Lz = 100, 100, 100
shape = nx, ny, nz = 10, 10, 10
coords, topo = grid(size=size, shape=shape, eshape='H27')
GlobalFrame = StandardFrame(dim=3)
pd = PointData(coords=coords, frame=GlobalFrame)
mesh = PolyData(pd, frame=GlobalFrame)

part1 = H27(topo=topo[:10], frames=GlobalFrame)
part2 = H27(topo=topo[10:-10], frames=GlobalFrame)
part3 = H27(topo=topo[-10:], frames=GlobalFrame)

mesh['A']['Part1'] = PolyData(cd=part1)
mesh['A']['Part2'] = PolyData(cd=part2)
mesh['A']['Part3'] = PolyData(cd=part3)

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
