"""
# Recipes for Mesh Generation
"""

# %% [markdown]
# ## Cylinder

# %%
from polymesh.recipes import cylinder

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
cyl = cylinder(shape, size)
cyl.plot(notebook=True, window_size=(800, 600))

# %%
cyl = cylinder(shape, size, regular=False)
cyl.plot(notebook=True, window_size=(800, 600))

# %%
cyl = cylinder(shape, size, voxelize=True)
cyl.plot(notebook=True, window_size=(800, 600))

# %%
from polymesh.topo.tr import H8_to_L2
import numpy as np
coords = cyl.coords()
topo = cyl.topology()
c = cyl.centers()[:, 2]
cm = (c.min() + c.max())/2
upper = np.where(c > cm)[0]
lower = np.where(c <= cm)[0]
coordsL2, topoL2 = H8_to_L2(coords, topo[upper])

# %%
from polymesh import PolyData
from polymesh.cells import L2, H8
pd = PolyData(coords=coordsL2)
pd['beams', 'L2'] = PolyData(topo=topoL2, celltype=L2)
pd['body', 'H8'] = PolyData(topo=topo[lower], celltype=H8)

# %%
pd.plot(notebook=True, window_size=(800, 600))

# %%
pd.topology()

# %%
pd.topology()[0, :]

# %%
pd.topology()[-1, :]

# %%
pd.topology()[-1, :].to_numpy()

# %%
from polymesh.pointdata import PointData

# %%
pd2 = PointData(coords=pd.coords(), z=pd.coords()[:, :2])

# %%
pd2.z.to_numpy()


