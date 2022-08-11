# -*- coding: utf-8 -*-
from typing import Union
import numpy as np
from numpy import ndarray

from .polydata import PolyData
from .tri.trimesh import TriMesh
from .cells import H8, TET4, T3
from .space import CartesianFrame
from .tri.triang import triangulate
from .topo import detach_mesh_bulk
from .extrude import extrude_T3_TET4
from .voxelize import voxelize_cylinder


__all__ = ['circular_disk', 'cylinder']


def circular_disk(nangles: int, nradii: int, rmin: float, rmax: float,
                  frame=None) -> TriMesh:
    """
    Returns the triangulation of a circular disk.

    Parameters
    ----------
    nangles : int
        Number of subdivisions in radial direction.

    nradii : int
        Number of subdivisions in circumferential direction.

    rmin : float
        Inner radius. Can be zero.

    rmax : float
        Outer radius.

    Returns
    -------
    TriMesh

    Examples
    --------
    >>> from dewloosh.mesh.recipes import circular_disk
    >>> mesh = circular_disk(120, 60, 5, 25) 

    """
    radii = np.linspace(rmin, rmax, nradii)
    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], nradii, axis=1)
    angles[:, 1::2] += np.pi / nangles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    nP = len(x)
    points = np.stack([x, y], axis=1)
    *_, triang = triangulate(points=points, backend='mpl')
    #triang = tri.Triangulation(x, y)
    # Mask off unwanted triangles.
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                             y[triang.triangles].mean(axis=1))
                    < rmin)
    triangles = triang.get_masked_triangles()
    points = np.stack((triang.x, triang.y, np.zeros(nP)), axis=1)
    points, triangles = detach_mesh_bulk(points, triangles)
    frame = CartesianFrame(dim=3) if frame is None else frame
    return TriMesh(points=points, triangles=triangles, celltype=T3, frame=frame)


def cylinder(shape, size:Union[tuple, float, int]=None, *args, 
             regular=True, voxelize=False, celltype=None, frame=None, 
             **kwargs) -> PolyData:
    """
    Returns the coordinates and the topology of cylinder as numpy arrays.
    
    Parameters
    ----------
    shape : tuple or int, Optional
        A 2-tuple or a float, describing the shape of the cylinder.
                
    size : Union[tuple, float, int], Optional
        Parameter controlling the density of the mesh. Default is None. 
        
        If `voxelize` is ``False``, ``size`` must be a tuple of three
        integers, describing the number of angular, radial, and vertical
        divions in this order.
        
        If `voxelize` is ``True`` and ``size`` is a ``float``, 
        the parameter controls the size of the individual voxels.
        
        If `voxelize` is ``True`` and ``size`` is an ``int``, 
        the parameter controls the size of the individual voxels
        according to :math:`edge \, length = (r_{ext} - r_{int})/shape`.
        
    regular : bool, Optional
        If ``True`` and ``voxelize`` is False, the mesh us a result of an extrusion
        applied to a trianglarion, and as a consequence it returns a more or 
        less regular mesh. Otherwise the cylinder is created from a surface
        trangulation using the ``tetgen`` package. Default is ``True``.

    voxelize : bool, Optional.
        If ``True``, the cylinder gets voxelized to a collection of H8 cells.
        In this case the size of a voxel can be controlled by specifying a 
        flaot or an integer as the second parameter ``size``.
        Default is ``False``.

    celltype
        Specifies the celltype to be used.
    
    Returns
    -------
    PolyData
    
    Examples
    --------
    >>> from dewloosh.mesh.recipes import cylinder
    >>> mesh = cylinder(120, 60, 5, 25)
        
    """
    if celltype is None:
        celltype = H8 if voxelize else TET4
    etype = None
    if isinstance(size, float) or isinstance(size, int):
        size = [size]
    if voxelize:
        regular = True
        etype = 'H8'
    radius, angle, h = shape
    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)
    etype = celltype.__label__ if etype is None else etype
    if voxelize:
        if isinstance(size[0], int):
            size_ = (radius[1] - radius[0]) / size[0]
        elif isinstance(size[0], float):
            size_ = size[0]
        coords, topo = voxelize_cylinder(radius=radius, height=h, size=size_)
    else:
        if regular:
            if etype == 'TET4':
                min_radius, max_radius = radius
                n_radii, n_angles, n_z = size
                points, triangles = circular_disk(
                    n_angles, n_radii, min_radius, max_radius)
                coords, topo = extrude_T3_TET4(points, triangles, h, n_z)
            else:
                raise NotImplementedError("Celltype not supported!")
        else:
            import tetgen
            import pyvista as pv
            (rmin, rmax), angle, h = shape
            n_radii, n_angles, n_z = size
            cyl = pv.CylinderStructured(center=(0.0, 0.0, h/2), direction=(0.0, 0.0, 1.0),
                                        radius=np.linspace(rmin, rmax, n_radii), height=h,
                                        theta_resolution=n_angles, z_resolution=n_z)
            cyl_surf = cyl.extract_surface().triangulate()
            tet = tetgen.TetGen(cyl_surf)
            tet.tetrahedralize(order=1, mindihedral=10,
                               minratio=1.1, quality=True)
            grid = tet.grid
            coords = np.array(grid.points).astype(float)
            topo = grid.cells_dict[10].astype(int)
    
    frame = CartesianFrame(dim=3) if frame is None else frame
    return PolyData(coords=coords, topo=topo, celltype=celltype, frame=frame)