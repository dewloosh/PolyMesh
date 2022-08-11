# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
from typing import MutableMapping, Union, Dict, List, Tuple
import awkward as ak
from awkward import Array as akarray
from scipy.sparse import csr_matrix as csr_scipy

from ...math.linalg.sparse import csr_matrix, JaggedArray
from ...math.arraysetops import unique2d
from ...math.array import count_cols

from .. space import PointCloud
from ..utils import explode_mesh_bulk
from ..config import __hasnx__

from .topoarray import TopologyArray
from .topodata import faces_TET4, faces_H8


if __hasnx__:
    import networkx as nx


__all__ = ['is_regular', 'regularize', 'count_cells_at_nodes', 'cells_at_nodes',
           'nodal_adjacency', 'unique_topo_data', 'remap_topo', 'detach_mesh_bulk',
           'rewire', 'detach', 'extract_tet_surface']


__cache = True
CoordsLike = Union[ndarray, PointCloud]
TopoLike = Union[ndarray, JaggedArray, akarray]
MappingLike = Union[ndarray, MutableMapping]
DoL = Dict[int, List[int]]


def flatten_line_mesh(coords: ndarray, topo: ndarray):
    raise NotImplementedError
    count = count_cells_at_nodes(topo)
    mess = np.where(count < 1)[0]
    leafs = np.where(count == 1)[0]
    forks = np.where(count > 2)[0]
    inters = np.where(count == 2)[0]
    nP, _ = coords.shape
    nE, nNE = topo.shape
    if nP == nE * nNE:
        _, t = coords, topo
    else:
        _, t = explode_mesh_bulk(coords, topo)
    t_ = np.sort(t, axis=1)[:, 0]
    order = np.argsort(t[:, 0])
    t_ = t_[order]
    return coords, order


def rewire(topo: TopoLike, imap: MappingLike, invert=False):
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.

    Parameters
    ----------
    topo : numpy.ndarray array or JaggedArray
        1d or 2d integer array representing topological data of a mesh.

    imap : MappingLike
        Inverse mapping on the index sets from global to local.   

    invert : bool, Optional
        If `True` the argument `imap` describes a local to global
        mapping and an inversion takes place. In this case, 
        `imap` must be a `numpy` array. Default is False.

    Returns
    -------
    TopoLike
        The same topology with the new numbering.

    """
    if invert:
        assert isinstance(imap, ndarray)
        imap = inds_to_invmap_as_dict(imap)
    if isinstance(topo, ndarray):
        if len(topo.shape) == 2:
            return remap_topo(topo, imap)
        elif len(topo.shape) == 1:
            return remap_topo_1d(topo, imap)
    elif isinstance(topo, TopologyArray):
        cuts, topo1d = topo.flatten(return_cuts=True)
        topo1d = remap_topo_1d(topo1d, imap)
        return TopologyArray(JaggedArray(topo1d, cuts=cuts))
    elif isinstance(topo, akarray):
        cuts = count_cols(topo)
        topo1d = ak.flatten(topo).to_numpy()
        topo1d = remap_topo_1d(topo1d, imap)
        return JaggedArray(topo1d, cuts=cuts)
    else:
        raise TypeError("Invalid topology with type <{}>".format(type(topo)))


@njit(nogil=True, parallel=True, cache=__cache)
def remap_topo(topo: ndarray, imap):
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.

    """
    nE, nNE = topo.shape
    res = np.zeros_like(topo)
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[iE, jNE] = imap[topo[iE, jNE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def remap_topo_1d(topo1d: ndarray, imap):
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.

    """
    N = topo1d.shape[0]
    res = np.zeros_like(topo1d)
    for i in prange(N):
        res[i] = imap[topo1d[i]]
    return res


def is_regular(topo: TopoLike) -> bool:
    """
    Returns True if the topology is regular, in the meaning
    that the smallest node index is zero, and every integer
    is represented up to the maximum index.
    """
    if isinstance(topo, ndarray):
        return topo.min() == 0 and len(np.unique(topo)) == topo.max() + 1
    elif isinstance(topo, akarray):
        return np.min(topo) == 0 and len(unique2d(topo)) == np.max(topo) + 1
    else:
        raise NotImplementedError


def regularize(topo: TopoLike) -> Tuple[TopoLike, ndarray]:
    """
    Returns a regularized topology and the unique indices.
    The returned topology array contains indices of the unique
    array.
    """
    if isinstance(topo, ndarray):
        unique, regular = np.unique(topo, return_inverse=True)
        regular = regular.reshape(topo.shape)
        return regular, unique
    elif isinstance(topo, akarray):
        unique, regular = unique2d(topo, return_inverse=True)
        return regular, unique
    else:
        raise NotImplementedError


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_reg_np_(topo: ndarray) -> ndarray:
    """
    Assumes a regular topology. Returns an array.
    """
    nE, nNE = topo.shape
    nN = topo.max() + 1
    count = np.zeros((nN), dtype=topo.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            count[topo[iE, jNE]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_reg_ak_(topo: akarray) -> ndarray:
    """
    Assumes a regular topology. Returns an array.
    """
    ncols = count_cols(topo)
    nE = len(ncols)
    nN = np.max(topo) + 1
    count = np.zeros((nN), dtype=topo.dtype)
    for iE in prange(nE):
        for jNE in prange(ncols[iE]):
            count[topo[iE, jNE]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_np_(topo: ndarray, nodeIDs: ndarray) -> Dict[int, int]:
    """
    Returns a dict{int : int} for the nodes in `nideIDs`.
    Assumes an irregular topology. The array `topo` must contain 
    indices relative to `nodeIDs`. If the topology is regular, 
    `nodeIDs == np.arange(topo.max() + 1)` is `True`.
    """
    nE, nNE = topo.shape
    count = dict()
    for i in range(len(nodeIDs)):
        count[nodeIDs[i]] = 0
    for iE in prange(nE):
        for jNE in prange(nNE):
            count[nodeIDs[topo[iE, jNE]]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_ak_(topo: akarray, nodeIDs: ndarray) -> Dict[int, int]:
    """
    Returns a dict{int : int} for the nodes in `nideIDs`.
    Assumes an irregular topology. The array `topo` must contain 
    indices relative to `nodeIDs`. If the topology is regular, 
    `nodeIDs == np.arange(topo.max() + 1)` is `True`.
    """
    ncols = count_cols(topo)
    nE = len(ncols)
    count = dict()
    for i in range(len(nodeIDs)):
        count[nodeIDs[i]] = 0
    for iE in prange(nE):
        for jNE in prange(ncols[iE]):
            count[nodeIDs[topo[iE, jNE]]] += 1
    return count


def count_cells_at_nodes(topo: TopoLike, regular=False):
    """
    Returns an array or a discionary, that counts connecting 
    elements at the nodes of a mesh.

    Parameters
    ----------
    topo : TopoLike
        2d numpy or awkward array describing the topoogy of a mesh.

    Returns
    -------
    count : np.ndarray(nN) or dict[int : int]
        Number of connecting elements for each node in a mesh.
    """
    if not regular:
        if isinstance(topo, ndarray):
            topo, nodeIDs = regularize(topo)
            return _count_cells_at_nodes_np_(topo, nodeIDs)
        elif isinstance(topo, akarray):
            topo, nodeIDs = regularize(topo)
            return _count_cells_at_nodes_ak_(topo, nodeIDs)
        else:
            raise NotImplementedError
    else:
        if isinstance(topo, ndarray):
            return _count_cells_at_nodes_reg_np_(topo)
        elif isinstance(topo, akarray):
            return _count_cells_at_nodes_reg_ak_(topo)
        else:
            raise NotImplementedError


def cells_at_nodes(topo: TopoLike, *args, frmt=None, assume_regular=False,
                   cellIDs=None, return_counts=False, **kwargs):
    """ 
    Returns data about element connectivity at the nodes of a mesh.

    Parameters
    ----------
    topo : numpy.ndarray array or JaggedArray
        A 2d array (either jagged or not) representing topological data of a mesh.

    frmt : str
        A string specifying the output format. Valid options are
        'jagged', 'csr', 'scipy-csr' and 'dicts'. 
        See below for the details on the returned object.   

    return_counts : bool
        Wether to return the numbers of connecting elements at the nodes
        as a numpy array. If format is 'raw', the
        counts are always returned irrelevant to this option.

    assume_regular : bool
        If the topology is regular, you can gain some speed with providing
        it as `True`. Default is `False`.

    cellIDs : numpy.ndarray
        Indices of the cells in `topo`. If nor `None`, format must be 'dicts'.
        Default is `None`.

    Returns
    -------
    If `return_counts` is `True`, the number of connecting elements for
    each node is returned as either a numpy array (if `cellIDs` is `None`) 
    or a dictionary (if `cellIDs` is not `None`). If format is 'raw', the
    counts are always returned.

    `frmt` = None
        
        counts : np.ndarray(nN) - numbers of connecting elements
        
        ereg : np.ndarray(nN, nmax) - indices of connecting elements
        
        nreg : np.ndarray(nN, nmax) - node indices with respect to the
                                        connecting elements
    where
    
        nN - is the number of nodes,
       
        nmax - is the maximum number of elements that meet at a node, that is
            count.max()  
        
    `frmt` = 'csr'
        
        counts(optionally) : np.ndarray(nN) - number of connecting elements
       
        csr : csr_matrix - sparse matrix in a numba-jittable csr format.
                            Column indices denote element indices, values
                            have the meaning of element node locations.                       
    
    `frmt` = 'scipy-csr'
        
        counts(optionally) : np.ndarray(nN) - number of connecting elements
        
        csr : csr_matrix - An instance of scipy.linalg.sparse.csr_matrix.
                            Column indices denote element indices, values
                            have the meaning of element node locations.
    `frmt` = 'dicts'
        
        counts(optionally) : np.ndarray(nN) - number of connecting elements
        
        ereg : numba Dict(int : int[:]) - indices of elements for each node
                                            index
        
        nreg : numba Dict(int : int[:]) - local indices of nodes in the
                                            connecting elements   
    
    `frmt` = 'jagged'
        
        counts(optionally) : np.ndarray(nN) - number of connecting elements
        
        ereg : JaggedArray - indices of elements for each node index
        
        nreg : JaggedArray - local indices of nodes in the connecting elements   

    """
    if cellIDs is not None:
        assert frmt == 'dicts', "If `cellIDs` is not None,"\
            + " output format must be 'dicts'."

    if not assume_regular:
        if is_regular(topo):
            nodeIDs = None
        else:
            topo, nodeIDs = regularize(topo)
    else:
        nodeIDs = None

    if nodeIDs is not None:
        assert frmt == 'dicts', "Only the format 'dicts' supports an irregular input!"

    if isinstance(topo, ndarray):
        counts, ereg, nreg = _cells_at_nodes_reg_np_(topo)
    elif isinstance(topo, akarray):
        counts, ereg, nreg = _cells_at_nodes_reg_ak_(topo)
    else:
        raise NotImplementedError

    frmt = '' if frmt is None else frmt

    if frmt in ['csr', 'csr-scipy']:
        data, indices, indptr, shape = \
            _nodal_cell_data_to_spdata_(counts, ereg, nreg)
        csr = csr_matrix(data=data, indices=indices,
                         indptr=indptr, shape=shape)
        if frmt == 'csr':
            csr = csr_matrix(data=data, indices=indices,
                             indptr=indptr, shape=shape)
        else:
            csr = csr_scipy((data, indices, indptr), shape=shape)
        if return_counts:
            return counts, csr
        return csr
    elif frmt == 'dicts':
        cellIDs = np.arange(len(topo)) if cellIDs is None else cellIDs
        nodeIDs = np.arange(len(counts)) if nodeIDs is None else nodeIDs
        ereg, nreg = _nodal_cell_data_to_dicts_(
            counts, ereg, nreg, cellIDs, nodeIDs)
        if return_counts:
            return counts, ereg, nreg
        return ereg, nreg
    elif frmt == 'jagged':
        data, indices, indptr, shape = \
            _nodal_cell_data_to_spdata_(counts, ereg, nreg)
        ereg = JaggedArray(indices, cuts=counts)
        nreg = JaggedArray(data, cuts=counts)
        if return_counts:
            return counts, ereg, nreg
        return ereg, nreg

    return counts, ereg, nreg


@njit(nogil=True, cache=__cache)
def _cells_at_nodes_reg_np_(topo: ndarray):
    """Returns arrays. Assumes a regular topology."""
    nE, nNE = topo.shape
    nN = topo.max() + 1
    count = _count_cells_at_nodes_reg_np_(topo)
    cmax = count.max()
    ereg = np.zeros((nN, cmax), dtype=topo.dtype)
    nreg = np.zeros((nN, cmax), dtype=topo.dtype)
    count[:] = 0
    for iE in range(nE):
        for jNE in range(nNE):
            ereg[topo[iE, jNE], count[topo[iE, jNE]]] = iE
            nreg[topo[iE, jNE], count[topo[iE, jNE]]] = jNE
            count[topo[iE, jNE]] += 1
    return count, ereg, nreg


@njit(nogil=True, cache=__cache)
def _cells_at_nodes_reg_ak_(topo: akarray):
    """Returns arrays. Assumes a regular topology."""
    ncols = count_cols(topo)
    nE = len(ncols)
    count = _count_cells_at_nodes_reg_ak_(topo)
    nN = np.max(topo) + 1
    cmax = count.max()
    ereg = np.zeros((nN, cmax), dtype=topo.dtype)
    nreg = np.zeros((nN, cmax), dtype=topo.dtype)
    count[:] = 0
    for iE in range(nE):
        for jNE in range(ncols[iE]):
            ereg[topo[iE, jNE], count[topo[iE, jNE]]] = iE
            nreg[topo[iE, jNE], count[topo[iE, jNE]]] = jNE
            count[topo[iE, jNE]] += 1
    return count, ereg, nreg


@njit(nogil=True, cache=__cache)
def _nodal_cell_data_to_dicts_(count: ndarray, ereg: ndarray,
                               nreg: ndarray, cellIDs: ndarray,
                               nodeIDs: ndarray) -> Tuple[Dict, Dict]:
    ereg_d = dict()
    nreg_d = dict()
    for i in range(len(count)):
        ereg_d[nodeIDs[i]] = cellIDs[ereg[i, : count[i]]]
        nreg_d[nodeIDs[i]] = nreg[i, : count[i]]
    return ereg_d, nreg_d


@njit(nogil=True, parallel=True, cache=__cache)
def _nodal_cell_data_to_spdata_(count: np.ndarray, ereg: np.ndarray,
                                nreg: np.ndarray) -> tuple:
    nE = ereg.max() + 1
    nN = len(count)
    N = np.sum(count)
    indices = np.zeros(N, dtype=count.dtype)
    data = np.zeros(N, dtype=count.dtype)
    indptr = np.zeros(nN+1, dtype=count.dtype)
    indptr[1:] = np.cumsum(count)
    for i in prange(nN):
        indices[indptr[i]: indptr[i+1]] = ereg[i, : count[i]]
        data[indptr[i]: indptr[i+1]] = nreg[i, : count[i]]
    shape = (nN, nE)
    return data, indices, indptr, shape


@njit(nogil=True, cache=__cache)
def _nodal_adjacency_as_dol_np_(topo: ndarray, ereg: DoL) -> DoL:
    """Returns nodal adjacency as a dictionary of lists."""
    res = dict()
    for iP in ereg:
        res[iP] = np.unique(topo[ereg[iP], :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _subtopo_1d_(topo1d: ndarray, cuts: ndarray, inds: ndarray,
                 indptr: ndarray) -> ndarray:
    nN = np.sum(cuts[inds])
    nE = len(inds)
    subindptr = np.zeros(nN+1, dtype=cuts.dtype)
    subindptr[1:] = np.cumsum(cuts[inds])
    subtopo1d = np.zeros(nN, dtype=topo1d.dtype)
    for iE in prange(nE):
        subtopo1d[subindptr[iE]: subindptr[iE+1]] = \
            topo1d[indptr[inds[iE]]: indptr[inds[iE]+1]]
    return subtopo1d


@njit(nogil=True, cache=__cache)
def _nodal_adjacency_as_dol_ak_(topo1d: ndarray, cuts: ndarray, ereg: DoL) -> DoL:
    """Returns nodal adjacency as a dictionary of lists."""
    res = dict()
    nN = len(cuts)
    indptr = np.zeros(nN+1, dtype=cuts.dtype)
    indptr[1:] = np.cumsum(cuts)
    for iP in ereg:
        res[iP] = np.unique(_subtopo_1d_(topo1d, cuts, ereg[iP], indptr))
    return res


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def dol_keys(dol: DoL) -> ndarray:
    nD = len(dol)
    res = np.zeros(nD, dtype=np.int64)
    c = 0
    for i in dol:
        res[c] = i
        c += 1
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dol_to_jagged_data(dol: DoL) -> Tuple[ndarray, ndarray]:
    nD = len(dol)
    keys = dol_keys(dol)
    widths = np.zeros(nD, dtype=np.int64)
    for i in prange(nD):
        widths[i] = len(dol[keys[i]])
    indptr = np.zeros(nD+1, dtype=np.int64)
    indptr[1:] = np.cumsum(widths)
    N = np.sum(widths)
    data1d = np.zeros(N, dtype=np.int64)
    for i in prange(nD):
        data1d[indptr[i]: indptr[i+1]] = dol[keys[i]]
    return widths, data1d


def detach(coords: CoordsLike, topo: TopoLike, inds: ndarray=None, return_indices=False):
    """
    Given a topology array and the coordinate array it refers to, 
    the function returns the coordinate array of the points involved 
    in the topology, and a new topology array, with indices referencing 
    the unique coordinate array.

    Parameters
    ----------
    coords : CoordsLike
        A 2d float array representing geometrical data of a mesh.
        If it is a `PointCloud` instance, indices may be included
        and parameter `inds` is obsolete.

    topo : TopoLike
        A 2d integer array (either jagged or not) representing topological 
        data of a mesh.

    inds : ndarray, Optional
        Global indices of the coordinates, in `coords`. If provided, the
        coordinates of node `j` of cell `i` is accessible as

        ``coords[imap[topo[i, j]]``,

        where `imap` is a mapping from local to global indices, and gets
        automatically generated from `inds`.  Default is None.

    return_indices : bool, Optional.
        If True, an index array for inverse mapping of point indices is returned.
        Default is False.
    
    Returns
    -------
    ndarray
        NumPy array similar to `coords`, but possibly with less entries.

    TopoLike
        Integer array representing the topology, with a good cahnce of 
        being jagged, depending on your input.

    """
    if isinstance(topo, ndarray):
        if inds is None and isinstance(coords, PointCloud):
            inds = coords.inds
        if isinstance(inds, ndarray):
            topo = rewire(topo, inds, True)
        return detach_mesh_bulk(coords, topo)
    elif hasattr(topo, '__array_function__'):
        if inds is None and isinstance(coords, PointCloud):
            inds = coords.inds
        if isinstance(inds, ndarray):
            topo = rewire(topo, inds, True)
        return detach_mesh_jagged(coords, topo)
    else:
        raise TypeError("Invalid topology with type <{}>".format(type(topo)))


def detach_mesh_jagged(coords: ndarray, topo: ndarray):
    """
    Given a topology array and the coordinate array it refers to, 
    the function returns the coordinate array of the points involved 
    in the topology, and a new topology array, with indices referencing 
    the new coordinate array.

    """
    inds = np.unique(topo)
    cuts, topo1d = topo.flatten(return_cuts=True)
    imap = inds_to_invmap_as_dict(inds)
    topo1d = remap_topo_1d(topo1d, imap)
    topo = JaggedArray(topo1d, cuts=cuts)
    return coords[inds, :], topo


@njit(nogil=True, cache=__cache)
def detach_mesh_bulk(coords: ndarray, topo: ndarray):
    """
    Given a topology array and the coordinate array it refers to, 
    the function returns the coordinate array of the points involved 
    in the topology, and a new topology array, with indices referencing 
    the new coordinate array.

    """
    inds = np.unique(topo)
    return coords[inds], remap_topo(topo, inds_to_invmap_as_dict(inds))


@njit(nogil=True, cache=__cache)
def inds_to_invmap_as_dict(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones.

    Parameters
    ----------
    inds : numpy.ndarray
        An array of global indices.

    Returns
    -------
    dict
        Mapping from global to local.
    """
    res = dict()
    for i in range(len(inds)):
        res[inds[i]] = i
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inds_to_invmap_as_array(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones
    as an array.

    Parameters
    ----------
    inds : numpy.ndarray
        An array of global indices.

    Returns
    -------
    numpy.ndarray
        Mapping from global to local.

    """
    res = np.zeros(inds.max() + 1, dtype=inds.dtype)
    for i in prange(len(inds)):
        res[inds[i]] = i
    return res


def nodal_adjacency(topo: TopoLike, *args, frmt=None,
                    assume_regular=False, **kwargs):
    """
    Returns nodal adjacency information of a mesh.

    Parameters
    ----------
    topo : numpy.ndarray array or JaggedArray
        A 2d array (either jagged or not) representing topological data of a mesh.

    frmt : str
        A string specifying the output format. Valid options are
        'jagged', 'csr' and 'scipy-csr'. See below for the details on the 
        returned object.

    assume_regular : bool
        If the topology is regular, you can gain some speed with providing
        it as `True`. Default is `False`.

    Notes
    -----
    1) You need `networkx` to be installed for most of the functionality here.

    2) A node is adjacent to itself.

    Returns
    -------
    `frmt` = None
        A dictionary of numpy arrays for each node.
    `frmt` = 'csr'
        csr_matrix - A sparse matrix in a numba-jittable csr format.
    `frmt` = 'scipy-csr'
        An instance of scipy.linalg.sparse.csr_matrix.
    `frmt` = 'nx'
        A networkx Graph.
    `frmt` = 'jagged'
        A JaggedArray instance.

    """
    frmt = '' if frmt is None else frmt
    ereg, _ = cells_at_nodes(topo, frmt='dicts', assume_regular=assume_regular)
    if isinstance(topo, ndarray):
        dol = _nodal_adjacency_as_dol_np_(topo, ereg)
    elif isinstance(topo, akarray):
        cuts, topo1d = JaggedArray(topo).flatten(return_cuts=True)
        dol = _nodal_adjacency_as_dol_ak_(topo1d.to_numpy(), cuts, ereg)
    if not __hasnx__ and frmt != 'jagged':
        errorstr = "`networkx` must be installed for format '{}'."
        raise ImportError(errorstr.format(frmt))
    if frmt == 'nx':
        return nx.from_dict_of_lists(dol)
    elif frmt == 'scipy-csr':
        G = nx.from_dict_of_lists(dol)
        return nx.to_scipy_sparse_matrix(G).tocsr()
    elif frmt == 'csr':
        G = nx.from_dict_of_lists(dol)
        csr = nx.to_scipy_sparse_matrix(G).tocsr()
        return csr_matrix(csr)
    elif frmt == 'jagged':
        cuts, data1d = dol_to_jagged_data(dol)
        return JaggedArray(data1d, cuts=cuts)
    return dol


def extract_tet_surface(topo: TopoLike):
    if isinstance(topo, ndarray):
        nNE = topo.shape[-1]
        if nNE == 4:
            faces3d = faces_TET4(topo)
            return unique_topo_data(faces3d)[0]
        elif nNE == 8:
            from .tr import Q4_to_T3
            faces3d = faces_H8(topo)
            unique_faces = unique_topo_data(faces3d)[0]
            return Q4_to_T3(None, unique_faces)[1]
            from .tr import H8_to_TET4
            _, topo = H8_to_TET4(None, topo)
            faces3d = faces_TET4(topo)
            return unique_topo_data(faces3d)[0]
    raise NotImplementedError


def unique_topo_data(topo3d: TopoLike):
    """
    Returns information about unique topological elements
    of a mesh. It can be used to return unique lines of a 2d
    mesh, unique faces of a 3d mesh, etc.

    Parameters
    ----------
    topo : numpy.ndarray 
        Topology array.

    Returns
    -------
    numpy.ndarray
        The sorted unique topolical entities as integer arrays 
        of node indices.

    numpy.ndarray
        Indices of the unique array, that can be used to 
        reconstruct `topo`. See the examples.

    Examples
    --------
    Find unique edges of a mesh of Q4 quadrilaterals

    >>> from dewloosh.mesh.rgrid import grid
    >>> from dewloosh.mesh.topo.topodata import edges_Q4

    >>> coords, topo = grid(size=(1, 1), shape=(10, 10), eshape='Q4')

    To get a 3d integer array listing all the edges of all quads:

    >>> edges3d = edges_Q4(topo)

    To find the unique edges of the mesh:

    >>> edges, edgeIDs = unique_topo_data(edges3d)

    Then, to reconstruct `edges3d`, do the following

    >>> edges3d_ = np.zeros_like(edges3d)
    >>> for i in range(edgeIDs.shape[0]):
    >>>     for j in range(edgeIDs.shape[1]):
    >>>         edges3d_[i, j, :] = edges[edgeIDs[i, j]]        
    >>> assert np.all(edges3d == edges3d_)
    True

    """
    """if isinstance(topo3d, ndarray):
        nE, nD, nN = topo3d.shape
        topo3d = topo3d.reshape((nE * nD, nN))
        
        i = np.argsort(topo3d, axis=1)
        topo = np.take_along_axis(topo3d, i, axis=1)
        #topo = np.sort(topo, axis=1)
        
        _, topoIDs = np.unique(topo, axis=0, return_inverse=True)
        return topo3d[topoIDs], topoIDs.reshape((nE, nD))"""
    if isinstance(topo3d, ndarray):
        nE, nD, nN = topo3d.shape
        topo3d = topo3d.reshape((nE * nD, nN))
        topo3d = np.sort(topo3d, axis=1)
        topo3d, topoIDs = np.unique(topo3d, axis=0, return_inverse=True)
        topoIDs = topoIDs.reshape((nE, nD))
        return topo3d, topoIDs
    elif isinstance(topo3d, JaggedArray):
        raise NotImplementedError


"""
def unique_lines(lines : np.ndarray):
    # ! IN PROGRESS
    raise NotImplementedError
    nE, nL, nNL = lines.shape
    lines = lines.reshape((nE * nL, nNL))
    lines = np.sort(lines, axis=1)
    lines, lineIDs = np.unique(lines[:, [0, -1]], axis=0, return_inverse=True)
    lineIDs = lineIDs.reshape((nE, nL))
    return lines, lineIDs


def unique_lines_of_areas(topo : np.ndarray, lpath : np.ndarray):
    # ! IN PROGRESS
    raise NotImplementedError
    return unique_lines(lines_of_areas_njit(topo, lpath))


@njit(nogil=True, parallel=True, cache=__cache)
def lines_of_areas_njit(topo : np.ndarray, lpaths : np.ndarray):
    # ! CHECK
    nE = topo.shape[0]
    nL, nNL = lpaths.shape
    lines = np.zeros((nE, nL, nNL), dtype=topo.dtype)
    for iE in prange(nE):
        for jL in prange(nL):
            for kL in prange(nNL):
                lines[iE, jL, kL] = topo[iE, lpaths[jL, kL]]
    return lines
"""
