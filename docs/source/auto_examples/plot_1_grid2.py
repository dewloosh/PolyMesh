"""
Grid Based Mesh Generation
==========================

"""

# %%
import numpy as np
from polymesh.grid import grid
from polymesh import PolyData, CartesianFrame
from polymesh.cells import H8, H27, Q9, Q4
size = Lx, Ly, Lz = 800, 600, 100
shape = nx, ny, nz = 8, 6, 2
xbins = np.linspace(0, Lx, nx+1)
ybins = np.linspace(0, Ly, ny+1)
zbins = np.linspace(0, Lz, nz+1)
bins = xbins, ybins, zbins
coords, topo = grid(bins=bins, eshape='H8')
pd = PolyData(coords=coords, topo=topo, celltype=H8)


# %% [markdown]
# The `PolyData` class delegates plotting-related jobs to `pyVista`. Call your objects `plot` method the same way you'd call a `pyVista` object:

# %%
pd.plot(off_screen=True, window_size = (600, 400))


# %%
coords, topo = grid(bins=bins, eshape='H27')
pd = PolyData(coords=coords, topo=topo, celltype=H27)
pd.plot(off_screen=True, window_size = (600, 400))

# %%
coords, topo = grid(bins=(xbins, ybins), eshape='Q4')
frame = CartesianFrame(dim=3)
pd = PolyData(coords=coords, topo=topo, celltype=Q4, frame=frame)
pd.plot(off_screen=True, window_size = (600, 400))

# %%
coords, topo = grid(bins=(xbins, ybins), eshape='Q9')
frame = CartesianFrame(dim=3)
pd = PolyData(coords=coords, topo=topo, celltype=Q9, frame=frame)
pd.plot(off_screen=True, window_size = (600, 400))

# %% [markdown]
# ## Voxelization

# %%
n_angles = 60
n_radii = 30
min_radius = 5
max_radius = 25
n_z = 20
h = 50
angle=1

shape = (min_radius, max_radius), angle, h
size = n_radii, n_angles, n_z

# %%
from polymesh.recipes import cylinder
cyl = cylinder(shape, size, voxelize=True)
cyl.plot(off_screen=True, window_size = (600, 400))


