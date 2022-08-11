# -*- coding: utf-8 -*-
# !TODO  handle decimation in all transformations, template : T6_to_T3
import numpy as np
from awkward import Array
from numba import njit, prange
from typing import Union, Sequence
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from numpy import ndarray, newaxis
from concurrent.futures import ThreadPoolExecutor

from ..tri.triutils import edges_tri
from ..utils import cells_coords
from .topodata import edgeIds_TET4, edgeIds_H8
from .topodata import edges_Q4, edges_H8, faces_H8
from .topo import unique_topo_data

__cache = True


__all__ = [
    'transform_topo',
    'T3_to_T6', 'T6_to_T3',
    'Q4_to_Q8',
    'Q4_to_Q9', 'Q9_to_Q4',
    'Q8_to_T3',
    'Q9_to_T6',
    'Q4_to_T3',
    'H8_to_Q4',
    'H8_to_H27',
    'H8_to_TET4'
]


DataLike = Union[ndarray, Sequence[ndarray]]


def transform_topo(topo: ndarray, path: ndarray, data: ndarray = None,
                   *args, MT=True, max_workers=4, **kwargs):
    nD = len(path.shape)
    if nD == 1:
        path = path.reshape(1, len(path))
    assert nD <= 2, "Path must be 1 or 2 dimensional."
    if data is None:
        return _transform_topo_(topo, path)
    else:
        if isinstance(data, Array):
            try:
                data = data.to_numpy()
            except Exception:
                raise TypeError("Invalid data type '{}'".format(data.__class__))
        if isinstance(data, ndarray):
            data = transform_topo_data(topo, data, path)
            return _transform_topo_(topo, path), data
        elif isinstance(data, Iterable):
            def foo(d): return transform_topo_data(topo, d, path)
            if MT:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    dmap = executor.map(foo, data)
            else:
                dmap = map(foo, data)
            return _transform_topo_(topo, path), list(dmap)


def transform_topo_data(topo: ndarray, data: ndarray, path: ndarray):
    if data.shape[:2] == topo.shape[:2]:
        # it is assumed that values are provided for each node of each cell
        res = repeat_cell_nodal_data(data, path)
    elif data.shape[0] == topo.shape[0]:
        # assume that data is constant over the elements
        res = np.repeat(data, path.shape[0], axis=0)
    else:
        raise NotImplementedError("Invalid data shape {}".format(data.shape))
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _transform_topo_(topo: ndarray, path: ndarray):
    nE = len(topo)
    nSub, nSubN = path.shape
    res = np.zeros((nSub * nE, nSubN), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * nSub
        for jE in prange(nSubN):
            for kE in prange(nSub):
                res[c + kE, jE] = topo[iE, path[kE, jE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat_cell_nodal_data(edata: ndarray, path: ndarray):
    nSub, nSubN = path.shape
    nE = edata.shape[0]
    res = np.zeros((nSub*nE, nSubN) + edata.shape[2:], dtype=edata.dtype)
    for i in prange(nE):
        ii = nSub*i
        for j in prange(nSub):
            jj = ii+j
            for k in prange(nSubN):
                res[jj, k] = edata[i, path[j, k]]
    return res


def T6_to_T3(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, decimate=True, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            if decimate:
                path = np.array([[0, 3, 5], [3, 1, 4],
                                 [5, 4, 2], [5, 3, 4]], dtype=topo.dtype)
            else:
                path = np.array([[0, 1, 2]], dtype=topo.dtype)
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def Q9_to_Q4(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array([[0, 4, 8, 7], [4, 1, 5, 8],
                             [8, 5, 2, 6], [7, 8, 6, 3]],
                            dtype=topo.dtype)
        elif isinstance(path, str):
            if path == 'grid':
                path = np.array([[0, 3, 4, 1], [3, 6, 7, 4],
                                 [4, 7, 8, 5], [1, 4, 5, 2]],
                                dtype=topo.dtype)
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray = None):
    if path is None:
        path = np.array([[0, 8, 2, 4, 5, 1], [0, 6, 8, 3, 7, 4]],
                        dtype=topo.dtype)
    return _Q9_to_T6(coords, topo, path)


@njit(nogil=True, parallel=True, cache=__cache)
def _Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray):
    nE = len(topo)
    topoT6 = np.zeros((2 * nE, 6), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * 2
        for jE in prange(6):
            topoT6[c, jE] = topo[iE, path[0, jE]]
            topoT6[c + 1, jE] = topo[iE, path[1, jE]]
    return coords, topoT6


def H8_to_TET4(coords: ndarray, topo: ndarray, data: DataLike = None,
               *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array([[0, 1, 2, 5], [0, 2, 3, 7], [4, 5, 7, 0],
                             [5, 6, 7, 2], [0, 2, 7, 5]], dtype=topo.dtype)
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def H8_to_Q4(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array([[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4],
                             [2, 3, 7, 6], [0, 3, 2, 1], [4, 5, 6, 7]],
                            dtype=topo.dtype)
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def TET4_to_L2(coords: ndarray, topo: ndarray, data: DataLike = None,
               *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[0] == 6, "Invalid shape!"
        assert path.shape[1] == 2, "Invalid shape!"
    else:
        if path is None:
            path = edgeIds_TET4()
        else:
            raise NotImplementedError("Invalid path!")
    if data is None:
        nE = len(topo)
        nSub, nSubN = path.shape
        topo = np.reshape(transform_topo(topo, path), (nE, nSub, nSubN))
        edges, _ = unique_topo_data(topo)
        return coords, edges
    else:
        raise NotImplementedError("Data conversion is not available here!")


def H8_to_L2(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[0] == 12, "Invalid shape!"
        assert path.shape[1] == 2, "Invalid shape!"
    else:
        if path is None:
            path = edgeIds_H8()
        else:
            raise NotImplementedError("Invalid path!")
    if data is None:
        nE = len(topo)
        nSub, nSubN = path.shape
        topo = np.reshape(transform_topo(topo, path), (nE, nSub, nSubN))
        edges, _ = unique_topo_data(topo)
        return coords, edges
    else:
        raise NotImplementedError("Data conversion is not available here!")


def Q4_to_T3(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            path = np.array([[0, 1, 2], [0, 2, 3]], dtype=topo.dtype)
        elif isinstance(path, str):
            if path == 'grid':
                path = np.array([[0, 2, 3], [0, 3, 1]], dtype=topo.dtype)
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def Q8_to_T3(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            path = np.array([
                [0, 4, 7],
                [4, 1, 5],
                [5, 2, 6],
                [6, 3, 7],
                [4, 6, 7],
                [4, 5, 6],
            ], dtype=topo.dtype)
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def L2_to_L3(coords: ndarray, topo: ndarray, order='ikj'):
    nP = len(coords)
    nodes, nodeIDs = unique_topo_data(topo[:, newaxis, :])
    new_coords = np.mean(coords[nodes], axis=1)
    new_topo = nodeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    if order == 'ikj':
        _buf = np.copy(topo[:, 1])
        topo[:, 1] = topo[:, 2]
        topo[:, 2] = _buf
    else:
        raise NotImplementedError
    return coords, topo


def T3_to_T6(coords: ndarray, topo: ndarray):
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_tri(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    return coords, topo


def Q4_to_Q8(coords: ndarray, topo: ndarray):
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo_res = np.hstack((topo, new_topo))
    coords_res = np.vstack((coords, new_coords))
    return coords_res, topo_res


def Q4_to_Q9(coords: ndarray, topo: ndarray):
    nP, nE = len(coords), len(topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new coords at element centers
    ecoords = cells_coords(coords, topo)
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.reshape(np.arange(nE) + nP, (nE, 1))
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_c))
    return coords_res, topo_res


def H8_to_H27(coords: ndarray, topo: ndarray):
    nP, nE = len(coords), len(topo)
    ecoords = cells_coords(coords, topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_H8(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new nodes on face centers
    faces, faceIDs = unique_topo_data(faces_H8(topo))
    coords_f = np.mean(coords[faces], axis=1)
    topo_f = faceIDs + nP
    nP += len(coords_f)
    # register new nodes in the cell centers
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.reshape(np.arange(nE) + nP, (nE, 1))
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_f, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_f, coords_c))
    return coords_res, topo_res
