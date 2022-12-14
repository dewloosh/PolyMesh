# -*- coding: utf-8 -*-
from typing import Union, MutableMapping, Iterable
from typing import Tuple, List, Callable

import numpy as np
from numpy import ndarray, squeeze
from sympy import Matrix, lambdify

from neumann.array import atleast1d, atleast2d, ascont
from neumann.utils import to_range_1d

from .celldata import CellData
from .utils import (jacobian_matrix_bulk, points_of_cells,
                    pcoords_to_coords_1d, cells_coords)
from .tri.triutils import area_tri_bulk
from .tet.tetutils import tet_vol_bulk
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from .topo import detach_mesh_bulk, rewire, TopologyArray

from .config import __haspyvista__
if __haspyvista__:
    import pyvista as pv

MapLike = Union[ndarray, MutableMapping]


class PolyCell(CellData):
    """
    A subclass of :class:`polymesh.celldata.CellData` as a base class 
    for all kinds of geometrical entities.

    """
    NNODE = None
    NDIM = None
    vtkCellType = None
    _face_cls_ = None
    shpfnc: Callable = None  # evaluator for shape functions
    shpmfnc: Callable = None  # evaluator for shape function matrices
    dshpfnc: Callable = None  # evaluator for shape function derivatives

    def __init__(self, *args, i: ndarray = None, **kwargs):
        if isinstance(i, ndarray):
            kwargs[self._dbkey_id_] = i
        super().__init__(*args, **kwargs)

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Ought to return local coordinates of the master element.

        Returns
        -------
        numpy.ndarray

        """
        raise NotImplementedError

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Ought to return the local coordinates of the center of the
        master element.

        Returns
        -------
        numpy.ndarray

        """
        raise NotImplementedError

    @classmethod
    def polybase(cls) -> Tuple[List]:
        """
        Ought to retrun the polynomial base of the master element. 

        Returns
        -------
        list
            A list of SymPy symbols.
        list
            A list of monomials.

        """
        raise NotImplementedError

    @classmethod
    def generate_shape_functions(cls):
        """
        Generates shape functions for the cell using SymPy.

        """
        nN = cls.NNODE
        nD = cls.NDIM
        nDOF = getattr(cls, 'NDOFN', 3)
        locvars, monoms = cls.polybase()
        monoms.pop(0)
        lcoords = cls.lcoords()
        if nD == 1:
            lcoords = np.reshape(lcoords, (nN, 1))

        def subs(lpos): return {v: lpos[i] for i, v in enumerate(locvars)}
        def mval(lpos): return [m.evalf(subs=subs(lpos)) for m in monoms]
        M = np.ones((nN, nN), dtype=float)
        M[:, 1:] = np.vstack([mval(loc) for loc in lcoords])
        coeffs = np.linalg.inv(M)
        monoms.insert(0, 1)
        shp = Matrix([np.dot(coeffs[:, i], monoms) for i in range(nN)])
        dshp = Matrix([[f.diff(m) for m in locvars] for f in shp])
        _shpf = lambdify([locvars], shp[:, 0].T, 'numpy')
        _dshpf = lambdify([locvars], dshp, 'numpy')

        def shpf(p:ndarray):
            """
            Evaluates the shape functions at multiple points in the 
            master domain.
            """
            #r = np.squeeze(_shpf([p[..., i] for i in range(nD)])).T
            #return ascont(r)
            r = np.stack([_shpf(p[i])[0] for i in range(len(p))])
            return ascont(r)
        
        def shpmf(p:ndarray):
            """
            Evaluates the shape function matrix at multiple points 
            in the master domain.
            """
            nP = p.shape[0]
            eye = np.eye(nDOF, dtype=float)
            shp = shpf(p)
            res = np.zeros((nP, nDOF, nN * nDOF), dtype=float)
            for iP in range(nP):
                for i in range(nN):
                    res[iP, :, i*nDOF: (i+1) * nDOF] = eye*shp[iP, i]
            return ascont(res)

        def dshpf(p:ndarray):
            """
            Evaluates the shape function derivatives at multiple points 
            in the master domain.
            """
            #r = np.squeeze(_dshpf([p[..., i] for i in range(nD)])).T
            #return ascont(np.swapaxes(r, -1, -2))
            r = np.stack([_dshpf(p[i]) for i in range(len(p))])
            return ascont(r)

        return shp, dshp, shpf, shpmf, dshpf

    @classmethod
    def shape_function_values(cls, pcoords: ndarray) -> ndarray:
        """
        Evaluates the shape functions at the specified locations.

        Parameters
        ----------
        pcoords : numpy.ndarray
            Locations of the evaluation points.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nD) where nP and nD are the number of 
            evaluation points and spatial dimensions.

        """
        if cls.NDIM == 3:
            if len(pcoords.shape)==1:
                pcoords = atleast2d(pcoords, front=True)
                return squeeze(cls.shpfnc(pcoords)).astype(float)
        return cls.shpfnc(pcoords).astype(float)
    
    @classmethod
    def shape_function_matrix(cls, pcoords: ndarray) -> ndarray:
        """
        Evaluates the shape function matrix at the specified locations.

        Parameters
        ----------
        pcoords : numpy.ndarray
            Locations of the evaluation points.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nDOF, nDOF * nNE) where nP, nDOF and nNE 
            are the number of evaluation points, degrees of freedom per node 
            and nodes per cell.

        """
        if cls.NDIM == 3:
            if len(pcoords.shape)==1:
                pcoords = atleast2d(pcoords, front=True)
                return squeeze(cls.shpmfnc(pcoords)).astype(float)
        return cls.shpmfnc(pcoords).astype(float)
    
    @classmethod
    def shape_function_derivatives(cls, pcoords: ndarray) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element. 

        Parameters
        ----------
        pcoords : numpy.ndarray
            Locations of the evaluation points.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE, nD), where nP, nNE and nD are 
            the number of evaluation points, nodes and spatial dimensions.

        """
        if cls.NDIM == 3:
            if len(pcoords.shape)==1:
                pcoords = atleast2d(pcoords, front=True)
                return squeeze(cls.dshpfnc(pcoords)).astype(float)
        return cls.dshpfnc(pcoords).astype(float)

    def measures(self, *args, **kwargs):
        """Ought to return measures for each cell in the database."""
        raise NotImplementedError

    def measure(self, *args, **kwargs):
        """Ought to return the net measure for the cells in the 
        database as a group."""
        return np.sum(self.measure(*args, **kwargs))

    def area(self, *args, **kwargs):
        """Returns the total area of the cells in the database. Only for 2d entities."""
        return np.sum(self.areas(*args, **kwargs))

    def areas(self, *args, **kwargs):
        """Ought to return the areas of the individuall cells in the database."""
        raise NotImplementedError

    def volume(self, *args, **kwargs):
        """Returns the volume of the cells in the database."""
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        """Ought to return the volumes of the individual cells in the database."""
        raise NotImplementedError

    def extract_surface(self, detach=False):
        """Extracts the surface of the mesh. Only for 3d meshes."""
        raise NotImplementedError

    def jacobian_matrix(self, *, dshp=None, ecoords=None, topo=None, **kwargs):
        """
        Returns the jacobian matrix.

        Parameters
        ----------
        dshp : numpy.ndarray
            3d array of shape function derivatives for the master cell.
        ecoords : numpy.ndarray, Optional
            3d array of nodal coordinates for all cells. 
            Either 'ecoords' or 'topo' must be provided.
        topo : numpy.ndarray, Optional
            2d integer topology array.
            Either 'ecoords' or 'topo' must be provided.

        Returns
        -------
        numpy.ndarray
            The 3d array of jacobian matrices for all the cells.

        """
        ecoords = self.local_coordinates(
            topo=topo) if ecoords is None else ecoords
        return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *args, jac=None, **kwargs):
        """
        Returns the jacobian determinant for one or more cells.

        Parameters
        ----------
        jac : numpy.ndarray, Optional
            One or more Jacobian matrices. Default is None.
        **kwargs : dict
            Forwarded to :func:`jacobian_matrix` if the jacobian
            is not provided by the parameter 'jac'.

        Returns
        -------
        float or numpy.ndarray
            Value of the Jacobian for one or more cells.

        See Also
        --------
        :func:`jacobian_matrix`

        """
        if jac is None:
            jac = self.jacobian_matrix(**kwargs)
        return np.linalg.det(jac)

    def points_of_cells(self, *args, **kwargs) -> ndarray:
        """
        Returns the points of the cells as a 3d float numpy array.       

        """
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return points_of_cells(coords, topo)

    def local_coordinates(self, *args, **kwargs) -> ndarray:
        """
        Returns local coordinates of the selection as a 3d float 
        numpy array.

        """
        frames = kwargs.get('frames', self.frames)
        topo = self.topology().to_numpy()
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        return points_of_cells(coords, topo, local_axes=frames)

    def coords(self, *args, **kwargs) -> ndarray:
        """
        Returns the coordinates of the cells in the database as a 3d 
        numpy array.

        """
        return self.points_of_cells(*args, **kwargs)

    def topology(self) -> TopologyArray:
        """
        Returns the numerical representation of the topology of 
        the cells.

        """
        key = self._dbkey_nodes_
        if key in self.fields:
            return TopologyArray(self.nodes)
        else:
            return None

    def rewire(self, imap: MapLike = None, invert:bool=False):
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping happens the
        following way:

        topology_new[old_index] = imap[topology_old[old_index]] 

        Parameters
        ----------
        imap : MapLike
            Mapping from old to new node indices (global to local).
        invert : bool, Optional
            If `True` the argument `imap` describes a local to global
            mapping and an inversion takes place. In this case, 
            `imap` must be a `numpy` array. Default is False.

        """
        if imap is None:
            imap = self.source().pointdata.id
        topo = self.topology().to_array().astype(int)
        topo = rewire(topo, imap, invert=invert).astype(int)
        self._wrapped[self._dbkey_nodes_] = topo


class PolyCell1d(PolyCell):
    """Base class for 1d cells"""

    NDIM = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lenth(self, *args, **kwargs):
        return np.sum(self.lengths(*args, **kwargs))

    def lengths(self, *args, **kwargs):
        raise NotImplementedError

    def areas(self, *args, **kwargs):
        raise NotImplementedError

    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))

    def measures(self, *args, **kwargs):
        return self.lengths(*args, **kwargs)

    # NOTE The functionality of `pcoords_to_coords_1d` needs to be generalized
    # for higher order cells.
    def points_of_cells(self, *args, points=None, cells=None,
                        target='global', rng=None, flatten=False,
                        **kwargs):
        if isinstance(target, str):
            assert target.lower() in ['global', 'g']
        else:
            raise NotImplementedError
        topo = kwargs.get('topo', self.topology().to_numpy())
        coords = kwargs.get('coords', None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        ecoords = points_of_cells(coords, topo)
        if points is None and cells is None:
            return ecoords

        # points or cells is not None
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
            ecoords = ecoords[cells]
            topo = topo[cells]
        else:
            cells = np.s_[:]

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        points, rng = to_range_1d(points, source=rng, target=[
            0, 1]).flatten(), [0, 1]
        datacoords = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)

        if not flatten:
            nE = ecoords.shape[0]
            nP = points.shape[0]
            datacoords = datacoords.reshape(
                nE, nP, datacoords.shape[-1])  # (nE, nP, nD)

        # values : (nE, nP, nDOF, nRHS) or (nE, nP * nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements
            data = datacoords
        elif isinstance(cells, Iterable):
            data = {c: datacoords[i] for i, c in enumerate(cells)}
        else:
            raise TypeError(
                "Invalid data type <> for cells.".format(type(cells)))

        return data


class PolyCell2d(PolyCell):
    """Base class for 2d cells"""

    NDIM = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))

    def areas(self, *args, **kwargs):
        raise NotImplementedError

    def to_triangles(self):
        raise NotImplementedError

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        topo_tri = self.to_triangles()
        areas = area_tri_bulk(cells_coords(coords, topo_tri))
        res = np.sum(areas.reshape(topo.shape[0], int(
            len(areas)/topo.shape[0])), axis=1)
        return np.squeeze(res)

    def area(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        return np.sum(self.areas(coords=coords, topo=topo))

    def volumes(self, *args, **kwargs):
        dbkey = self._dbkey_thickness_
        areas = self.areas(*args, **kwargs)
        if dbkey in self.fields:
            t = self.db[dbkey].to_numpy()
            return areas * t
        else:
            return areas

    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def measures(self, *args, **kwargs):
        return self.areas(*args, **kwargs)

    def local_coordinates(self, *args, **kwargs):
        ecoords = super(PolyCell2d, self).local_coordinates(*args, **kwargs)
        return ascont(ecoords[:, :, :2])

    def thickness(self, *args, **kwargs) -> ndarray:
        return self._wrapped[self._dbkey_thickness_].to_numpy()


class PolyCell3d(PolyCell):
    """Base class for 3d cells"""

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measures(self, *args, **kwargs):
        return self.volumes(*args, **kwargs)

    def to_tetrahedra(self) -> np.ndarray:
        raise NotImplementedError

    def to_vtk(self, detach=False):
        coords = self.container.root().coords()
        topo = self.topology().to_numpy()
        vtkid = self.__class__.vtkCellType
        if detach:
            ugrid = mesh_to_vtk(*detach_mesh_bulk(coords, topo), vtkid)
        else:
            ugrid = mesh_to_vtk(coords, topo, vtkid)
        return ugrid

    if __haspyvista__:
        def to_pv(self, detach=False) -> pv.UnstructuredGrid:
            return pv.wrap(self.to_vtk(detach=detach))

    def extract_surface(self, detach=False):
        pvs = self.to_pv(detach=detach).extract_surface(pass_pointid=True)
        s = pvs.triangulate().cast_to_unstructured_grid()
        topo = s.cells_dict[5]
        imap = s.point_data['vtkOriginalPointIds']
        topo = rewire(topo, imap)
        if detach:
            return s.points, topo
        else:
            return self.container.root().coords(), topo

    def boundary(self, detach=False):
        return self.surface(detach=detach)

    def volumes(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.container.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        topo_tet = self.to_tetrahedra()
        volumes = tet_vol_bulk(cells_coords(coords, topo_tet))
        res = np.sum(volumes.reshape(topo.shape[0], int(
            len(volumes) / topo.shape[0])), axis=1)
        return np.squeeze(res)
