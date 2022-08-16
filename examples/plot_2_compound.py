"""
Basic Mesh Composition and Plotting
===================================

"""

# %% [markdown]
# We are about to build a compound mesh by transforming parts of a voxlized cylinder into tetrahedra and lines. Then we do some plotting and review the basics of configuring plots.

# %% [markdown]
# The following block shows a basic example with minimal API usage and common `NumPy` operations:

# %%
from polymesh.space import CartesianFrame
from polymesh.recipes import cylinder
from polymesh.topo.tr import H8_to_L2, H8_to_TET4
from polymesh.topo import detach_mesh_bulk
from neumann.array import minmax
import numpy as np

frame = CartesianFrame(dim=3)

min_radius = 5
max_radius = 25
h = 50
angle = 1

shape = (min_radius, max_radius), angle, h
cyl = cylinder(shape, size=5., voxelize=True, frame=frame)

coords = cyl.coords()
topo = cyl.topology()
centers = cyl.centers()

cxmin, cxmax = minmax(centers[:, 0])
czmin, czmax = minmax(centers[:, 2])
cxavg = (cxmin + cxmax)/2
czavg = (czmin + czmax)/2
b_upper = centers[:, 2] > czavg
b_lower = centers[:, 2] <= czavg
b_left = centers[:, 0] < cxavg
b_right = centers[:, 0] >= cxavg
iL2 = np.where(b_upper & b_right)[0]
iTET4 = np.where(b_upper & b_left)[0]
iH8 = np.where(b_lower)[0]
_, tL2 = H8_to_L2(coords, topo[iL2])
_, tTET4 = H8_to_TET4(coords, topo[iTET4])
tH8 = topo[iH8]
cH8, tH8 = detach_mesh_bulk(coords, tH8)


# %%
E = 210000.0  # Young's modulus
nu = 0.3  # Poisson's ratio


# %%
from sigmaepsilon.solid import BeamSection, get_section
from sectionproperties.pre.pre import Material

steel = Material(
    name='Steel', elastic_modulus=E, poissons_ratio=nu, density=7.85e-6,
    yield_strength=250, color='grey'
)

section = get_section('CHS', D=1.0, t=0.1, n=64, material=steel)
beam = BeamSection(wrap=section)

A = np.array([
    [1, nu, nu, 0, 0, 0],
    [nu, 1, nu, 0, 0, 0],
    [nu, nu, 1, 0, 0, 0],
    [0., 0, 0, (1-nu)/2, 0, 0],
    [0., 0, 0, 0, (1-nu)/2, 0],
    [0., 0, 0, 0, 0, (1-nu)/2]]) * (E / (1-nu**2))


# %%
from sigmaepsilon.solid import Structure, LineMesh, PointData, FemMesh
from sigmaepsilon.solid.fem.cells import H8, TET4, B2

pd = PointData(coords, frame=frame)
mesh = FemMesh(pd, frame=frame)
mesh['lines', 'L2'] = LineMesh(
    B2(topo=tL2, frames=frame), 
    section=beam
    )
mesh['bodies', 'TET4'] = FemMesh(
    TET4(topo=tTET4, frames=frame), 
    model=A
    )
mesh['bodies', 'H8'] = FemMesh(
    PointData(cH8, frame=frame), 
    H8(topo=tH8, frames=frame),
    model=A
    )
structure = Structure(mesh=mesh)


# %% [markdown]
# ## Plotting with **PyVista**

# %% [markdown]
# Add some configuration to the `PyVista` plotting mechanism. This requires to think about
# a ``config_key``, that later we can use to point at our configuration. The configurations
# are stored in the ``config`` attribute of each block we are about to plot. For example,
# to set some basic properties:

# %%
mesh['lines', 'L2'].config['pyvista', 'plot', 'color'] = 'red'
mesh['lines', 'L2'].config['pyvista', 'plot', 'line_width'] = 2

mesh['bodies', 'TET4'].config['pyvista', 'plot', 'show_edges'] = True
mesh['bodies', 'TET4'].config['pyvista', 'plot', 'color'] = 'yellow'
mesh['bodies', 'TET4'].config['pyvista', 'plot', 'opacity'] = 0.8

mesh['bodies', 'H8'].config['pyvista', 'plot', 'show_edges'] = True
mesh['bodies', 'H8'].config['pyvista', 'plot', 'color'] = 'cyan'
mesh['bodies', 'H8'].config['pyvista', 'plot', 'opacity'] = 1.0


# %% [markdown]
# The following block shows how to refer to the previously set config. 

# %%
mesh.pvplot(notebook=True, jupyter_backend='static', cmap='plasma',
            window_size=(600, 400), config_key=('pyvista', 'plot'))


# %% [markdown]
# We can assign colors by setting numerical data on points or cells. All we have to do
# is to provide the keyword argument ``scalars`` to `PyVista`. One way to do that is to
# add these walues to the configuration of your block of choice. For example, to set random data on H8 hexagonals, do this:

# %%
mesh['bodies', 'H8'].config['pyvista', 'plot',
                            'scalars'] = np.random.rand(tH8.shape[0])
ncTET4 = mesh['bodies', 'TET4'].coords(from_cells=True).shape[0]
mesh['bodies', 'TET4'].config['pyvista', 'plot',
                              'scalars'] = 2*np.random.rand(ncTET4)
mesh['bodies', 'TET4'].config['pyvista', 'plot', 'opacity'] = 1.0
mesh.pvplot(notebook=True, jupyter_backend='static', window_size=(600, 400),
            config_key=('pyvista', 'plot'), cmap='plasma')


# %%
import pyvista as pv
plotter = pv.Plotter()
mesh['lines', 'L2'].config['pyvista', 'plot2', 'color'] = 'black'
mesh['lines', 'L2'].config['pyvista', 'plot2', 'line_width'] = 3
mesh.pvplot(plotter=plotter, config_key=('pyvista', 'plot2'))
plotter.show(jupyter_backend='static', window_size=(600, 400))


# %% [markdown]
# ### Background Plotting

# %% [markdown]
# This requires you to have `PyQt` and `pyvistaqt` installed, but it possible to plot
# using a widget:

# %%
"""from pyvistaqt import BackgroundPlotter
plotter = BackgroundPlotter(off_screen=False)
mesh.pvplot(plotter=plotter, config_key=('pyvista', 'plot2'))
plotter.render()
plotter.show()"""



