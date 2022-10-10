[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/PolyMesh/main?labpath=notebooks%5Cgrid.ipynb)
[![CircleCI](https://circleci.com/gh/dewloosh/PolyMesh.svg?style=shield)](https://circleci.com/gh/dewloosh/PolyMesh) 
[![Documentation Status](https://readthedocs.org/projects/polymesh/badge/?version=latest)](https://polymesh.readthedocs.io/en/latest/?badge=latest) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/PolyMesh.svg)](https://pypi.org/project/PolyMesh) 

# **PolyMesh** - A Python Library for Compound Meshes with Jagged Topology

> **Warning**
> This package is under active development and in a **beta stage**. Come back later, or star the repo to make sure you don’t miss the first stable release!

## **Documentation**

Click [here](https://PolyMesh.readthedocs.io/en/latest/) to read the documentation.

## **Installation**
This is optional, but we suggest you to create a dedicated virtual enviroment at all times to avoid conflicts with your other projects. Create a folder, open a command shell in that folder and use the following command

```console
>>> python -m venv venv_name
```

Once the enviroment is created, activate it via typing

```console
>>> .\venv_name\Scripts\activate
```

`PolyMesh` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.6:

```console
>>> pip install polymesh
```

## **Testing**

To run all tests, open up a console in the root directory of the project and type the following

```console
>>> python -m unittest
```

## **Dependencies**

must have 
  * `Numba`, `NumPy`, `SciPy`, `SymPy`, `awkward`

optional 
  * `networkx` : For graph analysis. Example application is the calculation of optimal elimination paths for the frontal solution technique. 
  
  * `pyvista` : PyVista is essential for visualizing in 3d. It is an easy access to `vtk`, which is a well-known member of the community. `PolyMesh` can translate complex
  models into `pyvista` or `vtk` objects any time, providing access to a host of 3rd party accessories.

  * `meshio` : For transformation to and from many mesh formats.

## **License**

This package is licensed under the MIT license.