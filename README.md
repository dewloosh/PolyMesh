# **PolyMesh** - A Python Library for Compound Meshes with Jagged Topologies

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/PolyMesh/main?labpath=notebooks%5Cgrid.ipynb)
[![CircleCI](https://circleci.com/gh/dewloosh/PolyMesh.svg?style=shield)](https://circleci.com/gh/dewloosh/PolyMesh)
[![Documentation Status](https://readthedocs.org/projects/polymesh/badge/?version=latest)](https://polymesh.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/PolyMesh.svg)](https://pypi.org/project/PolyMesh)
[![Python 3.7‒3.10](https://img.shields.io/badge/python-3.7%E2%80%923.10-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Warning**
> PolyMesh is in the early stages of it's lifetime, and some concepts may change in the future. If you want long-term stability, wait until version 1.0, which is planned to be released if the core concepts all seem to sit and the documentation covers all major concepts.

The [PolyMesh](https://PolyMesh.readthedocs.io/en/latest/) library aims to provide the tools to build and analyse meshes with complex topologies. Meshes can be built like a dictionary, using arbitarily nested layouts and then be translated to [VTK](https://vtk.org/) or  [PyVista](https://docs.pyvista.org/). For plotting, there is also support for [K3D](http://k3d-jupyter.org/), [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/python/).

The data model is built around [Awkward](https://awkward-array.org/doc/main/), which makes it possible to attach nested, variable-sized data to the points or the cells in a mesh, also providing interfaces to other popular libraries like [Pandas](https://vtk.org/) or [PyArrow](https://arrow.apache.org/docs/python/index.html). Implementations are fast as we rely on the vector math capabilities of [NumPy](https://numpy.org/doc/stable/index.html), while other computationally sensitive calculations are JIT-compiled using [Numba](https://numba.pydata.org/) where necessary.

Here and there we also use [NetworkX](https://networkx.org/documentation/stable/index.html#), [SciPy](https://scipy.org/), [SymPy](https://www.sympy.org/en/index.html) and [scikit-learn](https://scikit-learn.org/stable/).

## **Motivating example**

```python
from polymesh import PolyData, PointData, LineData
from polymesh.space import CartesianFrame
from polymesh.grid import Grid
from polymesh.cells import H8, TET4, L2
from polymesh.utils.topology import H8_to_TET4, H8_to_L2
from polymesh.utils.space import frames_of_lines
import numpy as np

size = 10, 10, 5
shape = 10, 10, 5
grid = Grid(size=size, shape=shape, eshape='H8')
grid.centralize()

coords = grid.coords()  # coordinates
topo = grid.topology()  # topology
centers = grid.centers()

b_left = centers[:, 0] < 0
b_right = centers[:, 0] >= 0
b_front = centers[:, 1] >= 0
b_back = centers[:, 1] < 0
iTET4 = np.where(b_left)[0]
iH8 = np.where(b_right & b_back)[0]
iL2 = np.where(b_right & b_front)[0]
_, tTET4 = H8_to_TET4(coords, topo[iTET4])
_, tL2 = H8_to_L2(coords, topo[iL2])
tH8 = topo[iH8]

# crate supporting pointcloud
frame = CartesianFrame(dim=3)
pd = PointData(coords=coords, frame=frame)
mesh = PolyData(pd, frame=frame)

# add tetrahedra
cdTET4 = TET4(topo=tTET4, frames=frame)
mesh['tetra'] = PolyData(cdTET4, frame=frame)

# add hexahedra
cdH8 = H8(topo=tH8, frames=frame)
mesh['hex'] = PolyData(cdH8, frame=frame)

# add lines
cdL2 = L2(topo=tL2, frames=frames_of_lines(coords, tL2))
mesh['line'] = LineData(cdL2, frame=frame)

# finalize the mesh and lock the layout
mesh.to_standard_form()
mesh.lock(create_mappers=True)
```

PolyMesh can also be used as a configuration tool for external plotting libraries:

```python
# configure tetratedra
mesh['tetra'].config['A', 'color'] = 'green'

# configure hexahedra
mesh['hex'].config['A', 'color'] = 'blue'

# configure lines
mesh['line'].config['A', 'color'] = 'red'
mesh['line'].config['A', 'line_width'] = 3
mesh['line'].config['A', 'render_lines_as_tubes'] = True

# plot with PyVista
mesh.plot(notebook=True, jupyter_backend='static', config_key=('A'),
          show_edges=True, window_size=(600, 480))
```

![ ](docs/source/_static/plot1.png)

Attaching data to the cells of the created blocks and plotting the results using PyVista:

```python
scalars_TET4 = 100*np.random.rand(len(cdTET4))
cdTET4.db['scalars'] = scalars_TET4

scalars_H8 = 100*np.random.rand(len(cdH8))
cdH8.db['scalars'] = scalars_H8

scalars_L2 = 100*np.random.rand(len(cdL2))
cdL2.db['scalars'] = scalars_L2
mesh['line'].config['B', 'render_lines_as_tubes'] = True
mesh['line'].config['B', 'line_width'] = 3

mesh.plot(notebook=True, jupyter_backend='static', config_key=('B'), 
          cmap='plasma', show_edges=True, window_size=(600, 480), 
          scalars='scalars')
```

![ ](docs/source/_static/plot2.png)

PolyMesh makes it easy to transfer data from the cells to the supporting point cloud:

```python
# this 'pulls' data from the cells
scalars = mesh.pointdata.pull('scalars') 
```

Then we can plot the smoothed data:

```python
mesh.plot(notebook=True, jupyter_backend='static', config_key=('A'),
          show_edges=True, window_size=(600, 480), scalars=scalars, 
          cmap='plasma')
```

![ ](docs/source/_static/plot3.png)

### Customizing the distribution mechanism

The smoothing procedure can be fine tuned using arbitrary weighting of cellular data. Define some scalar data on the cells and plot it using PyVista:

```python
cdTET4.db['scalars'] = np.full(len(cdTET4), -100)
cdH8.db['scalars'] = np.full(len(cdH8), 100)
cdL2.db['scalars'] = np.full(len(cdL2), 0)

mesh.plot(notebook=True, jupyter_backend='static', config_key=('B'), 
          cmap='jet', show_edges=True, window_size=(600, 480), 
          scalars='scalars')
```

![ ](docs/source/_static/plot4.png)

The default smoothing mechanism uses the volumes of the cells to determine nodal distribution factors.

```python
scalars = mesh.pd.pull('scalars')

mesh.plot(notebook=True, jupyter_backend='static', config_key=('A'),
          show_edges=True, window_size=(600, 480), scalars=scalars, 
          cmap='jet')
```

![ ](docs/source/_static/plot4a.png)

If you want you can give more presence to the hexahedral cells by increasing their weights:

```python
v = mesh.volumes()
idH8 = mesh['hex'].cd.id  # cell indices of hexahedra
v[idH8] *= 5  # 500% of original weight
ndf = mesh.nodal_distribution_factors(weights=v)
scalars = mesh.pd.pull('scalars', ndf=ndf)

mesh.plot(notebook=True, jupyter_backend='static', config_key=('A'),
          show_edges=True, window_size=(600, 480), scalars=scalars, 
          cmap='jet')
```

![ ](docs/source/_static/plot5.png)

or by decreasing them:

```python
v = mesh.volumes()
idH8 = mesh['hex'].cd.id  # cell indices of hexahedra
v[idH8] /= 5  # 20% of original weight
ndf = mesh.nodal_distribution_factors(weights=v)
scalars = mesh.pd.pull('scalars', ndf=ndf)

mesh.plot(notebook=True, jupyter_backend='static', config_key=('A'),
          show_edges=True, window_size=(600, 480), scalars=scalars, 
          cmap='jet')
```

![ ](docs/source/_static/plot6.png)

It can be observed how the colors change arounf the boundary of hexahedral cells.

Point-related data alsobe plotted using the K3D library:

```python
from k3d.colormaps import matplotlib_color_maps

cmap=matplotlib_color_maps.Jet
mesh.k3dplot(scalars=scalars, menu_visibility=False, cmap=cmap)
```

![ ](docs/source/_static/plot7.png)

## **Documentation**

The documentation is hosted on [ReadTheDocs](https://PolyMesh.readthedocs.io/en/latest/), where you can find more examples.

## **Installation**

PolyMesh can be installed from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install polymesh
```

## **License**

This package is licensed under the MIT license.
