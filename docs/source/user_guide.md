## **Motivating examples**

### Mesh assembly

One of the strongest sides of the library is mesh management. This example assembles a mesh of four separate bunnies using all kinds of transformations, each with their own separate pointcloud.

```python
from polymesh import PolyData
from polymesh.examples import download_bunny_coarse
import numpy as np
import pyvista as pv

mesh = PolyData()
mesh["bunny_1"] = download_bunny_coarse(tetra=False, read=True)
mesh["bunny_2"] = (
    mesh["bunny_1"]
    .spin("Space", [0, 0, np.pi/2], "XYZ", inplace=False)
    .move([0.2, 0, 0])
    )
mesh["bunny_3"] = (
    mesh["bunny_2"]
    .spin("Space", [0, 0, np.pi/2], "XYZ", inplace=False)
    .move([0.2, 0, 0])
    )
mesh["bunny_4"] = (
    download_bunny_coarse(tetra=True, read=True)
    .rotate("Space", [0, 0, 3*np.pi/2], "XYZ")
    .move([0.6, 0, 0])
    )
```

The following call centralizes the pointcloudes and revires the topologies.

```python
mesh.to_standard_form()
```

![ ](docs/source/_static/readme_1.png)

### Handling of jagged topologies

PolyMesh is able to handle the topologies of mixed meshes and return them as Awkward or NumPy arrays. In the previous example, one of the bunnies is a tetrahedral mesh, the others are surface triangulations.

```python
mesh.topology()
```

```console
[[0, 1, 2],
 [2, 1, 3],
 [4, 5, 6],
 [7, 5, 4],
 [4, 8, 7],
 [9, 10, 11],
 [12, 13, 14],
 [14, 15, 16],
 [15, 14, 13],
 [17, 13, 12],
 ...,
 [1182, 1163, 1715, 2285],
 [1162, 1163, 1182, 2285],
 [1121, 1983, 1641, 1074],
 [2189, 1232, 1613, 1585],
 [1642, 1121, 1667, 2286],
 [2122, 1945, 2175, 2103],
 [1739, 1925, 1742, 1740],
 [1191, 1748, 1749, 2287],
 [1191, 1202, 1748, 2287]]
--------------------------
type: 6769 * var * int32
```

```python
type(mesh.topology())
```

```console
polymesh.topoarray.TopologyArray
```

Similarly to NumPy arrays, a `TopologyArray` instance has a shape property which generalizes for jagged topologies nad coincides with NumPy for regular ones.

```python
mesh.topology().shape
```

```console
(6769, array([3, 3, 3, ..., 4, 4, 4], dtype=int64))
```

Calling `to_array` on a `TopologyArray` either returns an Awkward or a NumPy array.

```python
mesh.topology().to_array()
```

### Visualization

PolyMesh provides a mechanism to easily configure the blocks of a mesh to be plotted using PyVista:

```python
mesh["bunny_1"].config["plot"] = dict(color="red", opacity=0.9)
mesh["bunny_2"].config["plot"] = dict(color="green", opacity=0.9)
mesh["bunny_3"].config["plot"] = dict(color="blue", opacity=0.9)
mesh["bunny_4"].config["plot"] = dict(color="yellow", opacity=0.9)

plotter = mesh.plot(
    notebook=True, 
    config_key=["plot"], 
    return_plotter=True,
    theme=pv.themes.DarkTheme(),
    show_edges=False,
    lighting=True
)
plotter.camera.tight(padding=0.1, view="xz", negative=True)
plotter.show(jupyter_backend="static")
```

![ ](docs/source/_static/readme_2.png)

Values can be assigned to the cells

```python
for cb in mesh.cellblocks():
    n = len(cb.topology())
    cb.celldata["scalars"] = np.random.rand(n)

mesh["bunny_1"].config["plot"]["opacity"] = 1.0
mesh["bunny_2"].config["plot"]["opacity"] = 1.0
mesh["bunny_3"].config["plot"]["opacity"] = 1.0
mesh["bunny_4"].config["plot"]["opacity"] = 1.0
plotter = mesh.plot(
    notebook=True, 
    config_key=["plot"], 
    return_plotter=True,
    theme=pv.themes.DarkTheme(),
    show_edges=False,
    lighting=True,
    scalars="scalars",
    show_scalar_bar = False
)
plotter.camera.tight(padding=0.1, view="xz", negative=True)
plotter.show(jupyter_backend="static")
```

![ ](docs/source/_static/readme_3.png)

and to the points

```python
n = len(mesh.coords())
scalars = np.random.rand(n)
mesh.pd.db["scalars"] = scalars

plotter = mesh.plot(
    notebook=True, 
    config_key=["plot"], 
    return_plotter=True,
    theme=pv.themes.DarkTheme(),
    show_edges=False,
    lighting=True,
    scalars="scalars",
    show_scalar_bar = False
)
plotter.camera.tight(padding=0.1, view="xz", negative=True)
plotter.show(jupyter_backend="static")
```

![ ](docs/source/_static/readme_4.png)

### Passing data between points and cells

Values defined on the cells can also be aggregated to the nodes, creating a smoothing mechanism:

```python
plotter = mesh.plot(
    notebook=True, 
    config_key=["plot"], 
    return_plotter=True,
    theme=pv.themes.DarkTheme(),
    show_edges=False,
    lighting=True,
    scalars=mesh.pd.pull("scalars"),
    show_scalar_bar = False
)
plotter.camera.tight(padding=0.1, view="xz", negative=True)
plotter.show(jupyter_backend="static")
```

![ ](docs/source/_static/readme_5.png)

### Import and export

The heart of the database of a mesh is the combination of nested dictionaries equipped with Awkward records. Thanks to that, the data of a mesh can be easily converted to and from various data formats.

```python
from polymesh import PointData

mesh.pointdata.to_parquet("bunny.parquet")
mesh.pointdata = PointData.from_parquet("bunny.parquet")
```