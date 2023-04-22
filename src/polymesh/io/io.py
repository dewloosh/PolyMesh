# -*- coding: utf-8 -*-
from os.path import exists
from typing import Union, Any

import numpy as np
import meshio

from polymesh import PolyData
from polymesh.space import StandardFrame
from polymesh.utils.space import frames_of_surfaces, frames_of_lines

from ..cell import PolyCell
from ..vtkutils import PolyData_to_mesh
from ..helpers import meshio_to_celltype, vtk_to_celltype
from ..config import __haspyvista__

if __haspyvista__:
    import pyvista as pv
    pyVistaLike = Union[pv.PolyData, pv.PointGrid, pv.UnstructuredGrid]
else:
    pyVistaLike = Any


# TODO : read from image file with vtk
def input_to_mesh(*args, **kwargs) -> tuple:
    """
    Reads from several formats and returns the coordinates and the topology
    of a polygonal mesh. Every item in `args` is considered as a seprate
    candidate, but the function is able to handle several inputs at once.

    Parameters
    ----------
    args : Tuple
        Anything that can be converted into a polygonal mesh, that is

            1) a file supported by `meshio`

            2) an instance of `pyvista.PolyData` or `pyvista.UnstructuredGrid`

    Returns
    -------
    List[Tuple[array, Union[array, dict]]]
        A list of tuples, where the second item in each tuple is either a numpy
        array of node indices, or a dictionary of such arrays. The first item
        of the tuples is always a numpy cooridnate array.
    """
    candidate = kwargs.get("__candidate", None)
    if candidate is None:
        res = []
        for arg in args:
            res.append(input_to_mesh(__candidate=arg))
        return res

    coords, topo = None, None
    
    # read from file with meshio
    # TODO : inp2stl
    if isinstance(candidate, str):
        file_exists = exists(candidate)
        assert file_exists, "The file does not exist on this file system."
        mesh = meshio.read(
            candidate,  # string, os.PathLike, or a buffer/open file
            # file_format="stl",  # optional if filename is a path; inferred from extension
            # see meshio-convert -h for all possible formats
        )
        coords, topo = mesh.points, mesh.cells_dict

    # PyVista
    if coords is None and __haspyvista__:
        if isinstance(candidate, pv.PolyData):
            coords, topo = PolyData_to_mesh(candidate)
        elif isinstance(candidate, pv.UnstructuredGrid):
            coords, topo = candidate.points, candidate.cells_dict

    assert (
        coords is not None
    ), "Failed to read from this input, check the documentation!"
    return coords, topo


def from_meshio(mesh: meshio.Mesh) -> PolyData:
    GlobalFrame = StandardFrame(dim=3)

    coords = mesh.points
    polydata = PolyData(coords=coords, frame=GlobalFrame)

    for cb in mesh.cells:
        cd = None
        cbtype = cb.type
        celltype: PolyCell = meshio_to_celltype.get(cbtype, None)
        if celltype:
            topo = np.array(cb.data, dtype=int)
            
            NDIM = celltype.NDIM
            if NDIM == 1:
                frames = frames_of_lines(coords, topo)
            elif NDIM == 2:
                frames = frames_of_surfaces(coords, topo)
            elif NDIM == 3:
                frames = GlobalFrame
            
            cd = celltype(topo=topo, frames=frames)
            polydata[cbtype] = PolyData(cd, frame=GlobalFrame)
        else:
            msg = f"Cells of type '{cbtype}' are nut supported here."
            raise NotImplementedError(msg)

    return polydata


def from_pyvista(pvobj: pyVistaLike, numnode_to_celltype:dict=None) -> PolyData:
    
    if isinstance(pvobj, pv.PolyData):
        coords, topo = PolyData_to_mesh(pvobj)
        if isinstance(topo, dict):
            cells_dict = topo
        elif isinstance(topo, np.ndarray):
            assert isinstance(numnode_to_celltype, dict)
            ct = numnode_to_celltype[topo.shape[-1]]
            cells_dict = {ct.vtkCellType: topo}
    elif isinstance(pvobj, pv.UnstructuredGrid):
        coords = pvobj.points.astype(float)
        cells_dict = ugrid.cells_dict
    elif isinstance(pvobj, pv.PointGrid):
        ugrid = pvobj.cast_to_unstructured_grid()
        coords = pvobj.points.astype(float)
        cells_dict = ugrid.cells_dict
    else:
        try:
            ugrid = pvobj.cast_to_unstructured_grid()
            return from_pyvista(ugrid)
        except Exception:
            raise TypeError("Invalid inut type!")

    GlobalFrame = StandardFrame(dim=3)
    pd = PolyData(coords=coords, frame=GlobalFrame)  # this fails without a frame

    for vtkid, vtktopo in cells_dict.items():
        if vtkid in vtk_to_celltype:
            celltype = vtk_to_celltype[vtkid]
            
            NDIM = celltype.NDIM
            if NDIM == 1:
                frames = frames_of_lines(coords, topo)
            elif NDIM == 2:
                frames = frames_of_surfaces(coords, topo)
            elif NDIM == 3:
                frames = GlobalFrame
            
            cd = celltype(topo=vtktopo, frames=frames)
            pd[vtkid] = PolyData(cd, frame=GlobalFrame)
        else:
            msg = "The element type with vtkId <{}> is not jet" + "supported here."
            raise NotImplementedError(msg.format(vtkid))
        
    return pd
