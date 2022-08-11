# -*- coding: utf-8 -*-
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import numpy as np

from ....math.array import minmax

def plot_triangles_2d(coords, triangles, data):

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(800*px, 200*px))

    triobj = tri.Triangulation(coords[:, 0], coords[:, 2], triangles=triangles)

    ax.triplot(triobj, lw=0.5, color='white')

    refiner = tri.UniformTriRefiner(triobj)
    tri_refi, z_refi = refiner.refine_field(data, subdiv=3)

    dmin, dmax = minmax(data)
    levels = np.linspace(dmin, dmax, 10)

    triplot = ax.tricontourf(tri_refi, z_refi, levels=levels, cmap='jet')
    ax.tricontour(tri_refi, z_refi, levels=levels)

    ax.set_aspect('equal')
    fig.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(triplot, cax=cax)

    plt.show()