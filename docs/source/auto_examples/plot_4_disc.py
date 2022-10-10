"""
Corcular Disk
=============

"""

from polymesh.recipes import circular_disk

n_angles = 60
n_radii = 30
min_radius = 5
max_radius = 25

disk = circular_disk(n_angles, n_radii, min_radius, max_radius)

import pyvista as pv
from pyvista import themes

my_theme = themes.DarkTheme()
my_theme.color = 'red'
my_theme.lighting = False
my_theme.show_edges = True
my_theme.axes.box = True

pv.set_plot_theme(my_theme)

disk.pvplot(off_screen=True, window_size = (600, 400), theme=my_theme,
            jupyter_backend='static')