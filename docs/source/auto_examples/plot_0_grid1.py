"""
Plotting in 2d with `matplotlib`
================================

"""

# %% [markdown]
# ## Step 1 : Create a grid of quadrilaterals

# %%
from polymesh.grid import grid

gridparams = {
    'size' : (1200, 600),
    'shape' : (30, 15),
    'eshape' : (2, 2),
    'origo' : (0, 0),
    'start' : 0
}
coordsQ4, topoQ4 = grid(**gridparams)

# %% [markdown]
# ## Step 2 : Transform the mesh to triangles

# %%
from polymesh.topo.tr import Q4_to_T3
from polymesh.tri.trimesh import triangulate

points, triangles = Q4_to_T3(coordsQ4, topoQ4, path='grid')
triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

# %% [markdown]
# ## Step 3 : Plot the mesh

# %%
from dewloosh.mpl import triplot

triplot(triobj)

# %% [markdown]
# ## Step 4 : Plot the mesh with random data

# %% [markdown]
# Create a Hinton-plot with random data.

# %%
import numpy as np

data = np.random.rand(len(triangles))
triplot(triobj, hinton=True, data=data)

# %% [markdown]
# Plot the triangles with random cell data.

# %%
data = np.random.rand(len(triangles))
triplot(triobj, data=data)

# %% [markdown]
# Now plot the triangles with random point data and a 'bwr' colormap. Fot the different colormaps, see matplotlib's documentation.

# %% [markdown]
# _[Click here to see the built-in colormaps in matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html)_

# %%
data = np.random.rand(len(points))
triplot(triobj, data=data, cmap='bwr')

# %%
data = np.random.rand(len(points))
triplot(triobj, data=data, cmap='Set1', axis='off')

# %%
data = np.random.rand(len(points))
triplot(triobj, data=data, cmap='gnuplot', axis='off', ecolor='k', lw=0.8)


