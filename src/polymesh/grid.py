# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
from numpy import ndarray
from numba import njit, prange

from .topo import unique_topo_data, detach_mesh_bulk
from .topo.tr import transform_topo
from .utils import center_of_points, k_nearest_neighbours as knn, knn_to_lines
from .polydata import PolyData

__cache = True


__all__ = ['grid', 'gridQ4', 'gridQ9', 'gridH8', 'gridH27', 'knngridL2', 'Grid']


def grid(*args, size=None, shape=None, eshape=None, shift=None, start=0,
         bins=None, centralize=False, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Crates a 1d, 2d or 3d grid for different patterns.
    
    Parameters
    ----------

    size : tuple, Optional
        A 2-tuple, containg side lengths of a rectangular domain. \n
        Should be provided alongside `shape`.

    shape : tuple or int, Optional
        A 2-tuple, describing subdivisions along coordinate directions \n
        Should be provided alongside `size`.
        
    eshape : str or Tuple, Optional
        This specifies element shape. 
        Supported strings are thw following:
        
        'Q4' : 4-noded quadrilateral
           
        'Q9' : 9-noded quadrilateral
          
        'H8' : 8-noded hexagon
            
        'H27' : 27-noded hexagon
        
    shift: numpy.ndarray, Optional
        1d float array, specifying a translation.
                
    start : index, Optional
        Starting value for node numbering. Default is 0.
        
    bins : numpy.ndarray, Optional
        Numpy array or an iterable of numy arrays.
        
    centralize : bool, Optional
        If True, the returned coordinates are centralized.
           
    Notes
    -----
    1) The returned topology may not be compliant with vtk. If you want to use
    the results of this call to build a vtk model, you have to account for this.
    Optinally, you can use the dedicated grid generation routines of this module.
    
    2) If you"d rather get the result as a `PolyData`, use the `Grid` class.
    
    Returns
    -------
    numpy.ndarray
        A numpy float array of coordinates.
    
    numpy.ndarray
        A numpy integer array describing the topology.
    
    Examples
    --------
    Create a simple hexagonal mesh
        
    >>> size = 80, 60, 20
    >>> shape = 8, 6, 2
    >>> mesh = (coords, topo) = grid(size=size, shape=shape, eshape='H8')
    
    Create a mesh of 4-noded quadrilaterals
    
    >>> gridparams = {
    >>>     'size' : (1200, 600),
    >>>     'shape' : (30, 15),
    >>>     'eshape' : (2, 2),
    >>>     'origo' : (0, 0),
    >>>     'start' : 0
    >>> }
    >>> coordsQ4, topoQ4 = grid(**gridparams)
    
    The same mesh with 6-noded quadrialterals, 2 in x direction, 3 in y direction
    
    >>> gridparams['eshape'] = (2, 3)
    >>> coordsQ4, topoQ4 = grid(**gridparams)
    
    """
    if size is not None:
        nDime = len(size)
    elif bins is not None:
        nDime = len(bins)
    elif shape is not None:
        nDime = len(shape)

    if shift is None:
        shift = np.zeros(nDime)

    if eshape is None:
        eshape = np.full(nDime, 2, dtype=int)
    elif isinstance(eshape, str):
        if eshape == 'Q4':
            return gridQ4(*args, size=size, shape=shape, shift=shift,
                           start=start, centralize=centralize, bins=bins, **kwargs)
        elif eshape == 'Q9':
            return gridQ9(*args, size=size, shape=shape, shift=shift,
                           start=start, centralize=centralize, bins=bins, **kwargs)
        elif eshape == 'H8':
            return gridH8(*args, size=size, shape=shape, shift=shift,
                           start=start, centralize=centralize, bins=bins, **kwargs)
        elif eshape == 'H27':
            return gridH27(*args, size=size, shape=shape, shift=shift,
                            start=start, centralize=centralize, bins=bins, **kwargs)
        else:
            raise NotImplementedError
    
    if bins is not None:
        if len(bins) == 2:
            coords, topo = grid_2d_bins(bins[0], bins[1], eshape, shift, start)
        elif len(bins) == 3:
            coords, topo = grid_3d_bins(bins[0], bins[1], bins[2], eshape, shift, start)
        else:
            raise NotImplementedError
    else:
        assert isinstance(eshape, tuple)
        if shape is None:
            shape = np.full(nDime, 1, dtype=int)
        coords, topo = rgridMT(size, shape, eshape, shift, start)
    
    if centralize:
        center = center_of_points(coords)
        coords[:, 0] -= center[0]
        coords[:, 1] -= center[1]
        if center.shape[0] > 2:
            coords[:, 2] -= center[2]
    
    return coords, topo


def gridQ4(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to Q4 elements.
    It returns a topology with vtk-compliant node numbering.
    
    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
       
    Examples
    --------    
    Creating a mesh  a mesh of 4-noded quadrilaterals
    
    >>> gridparams = {
    >>>     'size' : (1200, 600),
    >>>     'shape' : (30, 15),
    >>>     'origo' : (0, 0),
    >>>     'start' : 0
    >>> }
    >>> coordsQ4, topoQ4 = gridQ4(**gridparams)
        
    """
    coords, topo = grid(*args, eshape=(2, 2), **kwargs)
    path = np.array([0, 2, 3, 1], dtype=int)
    return coords, transform_topo(topo, path)


def gridQ9(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to Q9 elements.
    It returns a topology with vtk-compliant node numbering.
    
    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
            
    """
    coords, topo = grid(*args, eshape=(3, 3), **kwargs)
    path = np.array([0, 6, 8, 2, 3, 7, 5, 1, 4], dtype=int)
    return coords, transform_topo(topo, path)


def gridH8(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to H8 elements.
    It returns a topology with vtk-compliant node numbering.
    
    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
            
    """
    coords, topo = grid(*args, eshape=(2, 2, 2), **kwargs)
    path = np.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=int)
    return coords, transform_topo(topo, path)


def gridH27(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to H27 elements.
    It returns a topology with vtk-compliant node numbering.
    
    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
            
    """
    coords, topo = grid(*args, eshape=(3, 3, 3), **kwargs)
    path = np.array([0, 18, 24, 6, 2, 20, 26, 8, 9, 21, 15, 3, 11, 23, 17, 5,
                     1, 19, 25, 7, 4, 22, 10, 16, 12, 14, 13], dtype=int)
    return coords, transform_topo(topo, path)


@njit(nogil=True, cache=__cache)
def _rgridST(size, shape, eshape, shift, start=0):
    """
    Legacy code for single-thread implementation, for educational purpose only. 
    Use rgridMT.
    """
    nDime = len(size)

    if nDime == 1:
        lX = size
        ndivX = shape
        nNodeX = eshape
        nX = ndivX * (nNodeX - 1) + 1
        dX = lX / ndivX
        ddX = dX / (nNodeX - 1)
        numCell = ndivX
        numPoin = nX
        numNode = nNodeX
    elif nDime == 2:
        lX, lY = size
        ndivX, ndivY = shape
        nNodeX, nNodeY = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        numCell = ndivX * ndivY
        numPoin = nX * nY
        numNode = nNodeX * nNodeY
    elif nDime == 3:
        lX, lY, lZ = size
        ndivX, ndivY, ndivZ = shape
        nNodeX, nNodeY, nNodeZ = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        nZ = ndivZ * (nNodeZ - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        dZ = lZ / ndivZ
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        ddZ = dZ / (nNodeZ - 1)
        numCell = ndivY * ndivX * ndivZ
        numPoin = nX * nY * nZ
        numNode = nNodeX * nNodeY * nNodeZ

    # set up nodal coordinates
    coords = np.zeros((numPoin, nDime))
    topo = np.zeros((numCell, numNode), dtype=np.int64)
    elem = -1
    if nDime == 1:
        for i in range(1, ndivX+1):
            elem += 1
            ne = -1
            n = (nNodeX-1) * (i-1)
            for j in range(1, nNodeX+1):
                n += 1
                coords[n-1, 0] = shift + dX * (i-1) + ddX * (j-1)
                ne += 1
                topo[elem, ne] = n
    elif nDime == 2:
        for i in range(1, ndivX + 1):
            for j in range(1, ndivY + 1):
                elem += 1
                ne = -1
                for k in range(1, nNodeX + 1):
                    for m in range(1, nNodeY + 1):
                        n = ((nNodeX - 1) * (i - 1) + k - 1) * nY + \
                            (nNodeY - 1) * (j - 1) + m
                        coords[n - 1, 0] = shift[0] + dX * (i - 1) + \
                            ddX * (k - 1)
                        coords[n - 1, 1] = shift[1] + dY * (j - 1) + \
                            ddY * (m - 1)
                        ne += 1
                        topo[elem, ne] = n
    elif nDime == 3:
        for i in range(1, ndivX+1):
            for j in range(1, ndivY+1):
                for k in range(1, ndivZ+1):
                    elem += 1
                    ne = -1
                    for m in range(1, nNodeX+1):
                        for q in range(1, nNodeY+1):
                            for p in range(1, nNodeZ+1):
                                n = (((nNodeX-1)*(i-1) + m-1)*nY +
                                     (nNodeY-1)*(j-1) + q-1)*nZ + \
                                    (nNodeZ-1)*(k-1) + p
                                coords[n-1, 0] = shift[0] + dX * (i-1) + \
                                    ddX * (m-1)
                                coords[n-1, 1] = shift[1] + dY * (j-1) + \
                                    ddY * (q-1)
                                coords[n-1, 2] = shift[2] + dZ * (k-1) + \
                                    ddZ * (p-1)
                                ne += 1
                                topo[elem, ne] = n
    start -= 1
    return coords, topo + start


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def rgridMT(size, shape, eshape, shift, start=0):
    nDime = len(size)
    if nDime == 1:
        lX = size[0]
        ndivX = shape
        nNodeX = eshape
        nX = ndivX * (nNodeX - 1) + 1
        dX = lX / ndivX
        ddX = dX / (nNodeX - 1)
        numCell = ndivX
        numPoin = nX
        numNode = nNodeX
    elif nDime == 2:
        lX, lY = size
        ndivX, ndivY = shape
        nNodeX, nNodeY = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        numCell = ndivX * ndivY
        numPoin = nX * nY
        numNode = nNodeX * nNodeY
    elif nDime == 3:
        lX, lY, lZ = size
        ndivX, ndivY, ndivZ = shape
        nNodeX, nNodeY, nNodeZ = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        nZ = ndivZ * (nNodeZ - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        dZ = lZ / ndivZ
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        ddZ = dZ / (nNodeZ - 1)
        numCell = ndivY * ndivX * ndivZ
        numPoin = nX * nY * nZ
        numNode = nNodeX * nNodeY * nNodeZ

    # set up nodal coordinates
    coords = np.zeros((numPoin, nDime))
    topo = np.zeros((numCell, numNode), dtype=np.int64)
    elem = -1
    if nDime == 1:
        for i in prange(1, ndivX+1):
            elem = i - 1
            for j in prange(1, nNodeX+1):
                n = (nNodeX - 1) * (i - 1) + j
                ne = j - 1
                coords[n - 1, 0] = shift[0] + dX * (i - 1) + ddX * (j - 1)
                topo[elem, ne] = n
    elif nDime == 2:
        for i in prange(1, ndivX + 1):
            for j in prange(1, ndivY + 1):
                elem = (i - 1) * ndivY + j - 1
                for k in prange(1, nNodeX + 1):
                    for m in prange(1, nNodeY + 1):
                        n = ((nNodeX - 1) * (i - 1) + k - 1) * nY + \
                            (nNodeY - 1) * (j - 1) + m
                        ne = (k - 1) * nNodeY + m - 1
                        coords[n - 1, 0] = shift[0] + dX * (i - 1) + \
                            ddX * (k - 1)
                        coords[n - 1, 1] = shift[1] + dY * (j - 1) + \
                            ddY * (m - 1)
                        topo[elem, ne] = n
    elif nDime == 3:
        for i in prange(1, ndivX+1):
            for j in prange(1, ndivY+1):
                for k in prange(1, ndivZ+1):
                    elem = (i - 1) * ndivY * ndivZ + (j - 1) * ndivZ + k - 1
                    for m in prange(1, nNodeX+1):
                        for q in prange(1, nNodeY+1):
                            for p in prange(1, nNodeZ+1):
                                n = (((nNodeX - 1) * (i - 1) + m - 1) * nY +
                                     (nNodeY - 1) * (j - 1) + q - 1) * nZ + \
                                    (nNodeZ - 1) * (k - 1) + p
                                ne = (m - 1) * nNodeY * nNodeZ + \
                                    (q - 1) * nNodeZ + p - 1
                                coords[n - 1, 0] = shift[0] + dX * (i - 1) + \
                                    ddX * (m - 1)
                                coords[n - 1, 1] = shift[1] + dY * (j - 1) + \
                                    ddY * (q - 1)
                                coords[n - 1, 2] = shift[2] + dZ * (k - 1) + \
                                    ddZ * (p - 1)
                                topo[elem, ne] = n
    start -= 1
    return coords, topo + start


@njit(nogil=True, parallel=True, cache=__cache)
def grid_2d_bins(xbins, ybins, eshape, shift, start=0):
       
    # size
    lX = xbins.max() - xbins.min()
    lY = ybins.max() - ybins.min()

    # shape of cells
    nEX = len(xbins) - 1
    nEY = len(ybins) - 1

    # number of cells
    nC = nEX * nEY

    # number of nodes
    nNEX, nNEY = eshape
    nNE = nNEX * nNEY
    nNX = nEX * (nNEX - 1) + 1
    nNY = nEY * (nNEY - 1) + 1
    nN = nNX * nNY
    dX = lX / nEX
    dY = lY / nEY
    ddX = dX / (nNEX - 1)
    ddY = dY / (nNEY - 1)

    # create grid
    coords = np.zeros((nN, 2))
    topo = np.zeros((nC, nNE), dtype=np.int64)
    for i in prange(nEX):
        for j in prange(nEY):
            iE = i * nEY + j
            for p in prange(nNEX):
                for q in prange(nNEY):
                    n = ((nNEX - 1) * i + p) * nNY + (nNEY - 1) * j + q
                    coords[n, 0] = shift[0] + xbins[i] + ddX * p
                    coords[n, 1] = shift[1] + ybins[j] + ddY * q
                    iNE = p * nNEY + q
                    topo[iE, iNE] = n
    return coords, topo + start


@njit(nogil=True, parallel=True, cache=__cache)
def grid_3d_bins(xbins, ybins, zbins, eshape, shift, start=0):
    # size
    lX = xbins.max() - xbins.min()
    lY = ybins.max() - ybins.min()
    lZ = zbins.max() - zbins.min()

    # shape of cells
    nEX = len(xbins) - 1
    nEY = len(ybins) - 1
    nEZ = len(zbins) - 1

    # number of cells
    nC = nEX * nEY * nEZ

    # number of nodes
    nNEX, nNEY, nNEZ = eshape
    nNE = nNEX * nNEY * nNEZ
    nNX = nEX * (nNEX - 1) + 1
    nNY = nEY * (nNEY - 1) + 1
    nNZ = nEZ * (nNEZ - 1) + 1
    nN = nNX * nNY * nNZ
    dX = lX / nEX
    dY = lY / nEY
    dZ = lZ / nEZ
    ddX = dX / (nNEX - 1)
    ddY = dY / (nNEY - 1)
    ddZ = dZ / (nNEZ - 1)

    # create grid
    coords = np.zeros((nN, 3), dtype=xbins.dtype)
    topo = np.zeros((nC, nNE), dtype=np.int64)
    for i in prange(nEX):
        for j in prange(nEY):
            for k in prange(nEZ):
                iE = i * nEZ * nEY + j * nEZ + k
                for p in prange(nNEX):
                    for q in prange(nNEY):
                        for r in prange(nNEZ):
                            n = (((nNEX - 1) * i + p) * nNY +
                                 (nNEY - 1) * j + q) * nNZ + \
                                (nNEZ - 1) * k + r
                            coords[n, 0] = shift[0] + xbins[i] + ddX * p
                            coords[n, 1] = shift[1] + ybins[j] + ddY * q
                            coords[n, 2] = shift[2] + zbins[k] + ddZ * r
                            iNE = p * nNEZ * nNEY + q * nNEZ + r
                            topo[iE, iNE] = n
    return coords, topo + start


def knngridL2(*args, max_distance=None, k=3, **kwargs):
    X, _ = grid(*args, **kwargs)
    i = knn(X, X, k=k, max_distance=max_distance)
    T, _ = unique_topo_data(knn_to_lines(i))
    di = T[:, 0] - T[:, -1]
    inds = np.where(di != 0)[0] 
    return detach_mesh_bulk(X, T[inds, :])


class Grid(PolyData):
    """
    A class to generate meshes based on grid-like data. All input
    arguments are forwarded to ``grid``.
    
    Examples
    --------
    >>> size = 80, 60, 20
    >>> shape = 8, 6, 2
    >>> grid = Grid(size=size, shape=shape, eshape='H8')
    
    """
    
    def __init__(self, *args, celltype=None, frame=None, 
                 eshape=None, **kwargs):
        # parent class handles pointdata and celldata creation
        coords, topo = grid(*args, eshape=eshape, **kwargs)
        super().__init__(*args, coords=coords, topo=topo, 
                         celltype=celltype, frame=frame, **kwargs)
        