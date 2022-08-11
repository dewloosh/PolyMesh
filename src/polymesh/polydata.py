# -*- coding: utf-8 -*-
from copy import copy, deepcopy
from typing import Union, Hashable, Collection, Iterable
from numpy import ndarray
import numpy as np
from awkward import Array as akarray


from ..core import DeepDict
from ..core.tools import suppress
from ..math.linalg.sparse import JaggedArray
from ..math.linalg import Vector, ReferenceFrame as FrameLike
from ..math.linalg.vector import VectorBase
from ..math.array import atleastnd

from .topo.topo import inds_to_invmap_as_dict, remap_topo_1d, extract_tet_surface
from .space import CartesianFrame, PointCloud
from .utils import cells_coords, cells_around, cell_center_bulk
from .utils import k_nearest_neighbours as KNN
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk, PolyData_to_mesh
from .cells import (
    T3 as Triangle,
    Q4 as Quadrilateral,
    H8 as Hexahedron,
    H27 as TriquadraticHexaHedron,
    Q9,
    TET10
)
from .polyhedron import Wedge
from .utils import index_of_closest_point, nodal_distribution_factors
from .topo import regularize, nodal_adjacency, detach_mesh_bulk, cells_at_nodes
from .topo.topoarray import TopologyArray
from .pointdata import PointData
from .celldata import CellData
from .base import PolyDataBase
from .cell import PolyCell

from .config import __hasvtk__, __haspyvista__, __hask3d__, __hasmatplotlib__
if __hasvtk__:
    import vtk
if __hask3d__:
    import k3d
if __hasmatplotlib__:
    import matplotlib as mpl

NoneType = type(None)

if __haspyvista__:
    import pyvista as pv
    from pyvista import themes
    pyVistaLike = Union[pv.PolyData, pv.UnstructuredGrid]
else:
    pyVistaLike = NoneType


VectorLike = Union[Vector, ndarray]
TopoLike = Union[ndarray, JaggedArray, akarray, TopologyArray]


__all__ = ['PolyData']


class PolyData(PolyDataBase):
    """
    A class to handle complex polygonal meshes.

    The `PolyData` class is arguably the most important class in the library 
    and a backbone of all mesh classes. 

    The implementation is based on the `awkward` library, which provides 
    memory-efficient, numba-jittable data classes to deal with dense, sparse, 
    complete or incomplete data. These data structures are managed in pure
    Python by the `DeepDict` class.

    The class accepts several kinds of inputs, allowing for a wide range of
    possible use cases. The fastes way to create a PolyData is from predefined 
    `PointData` and `CellData` instances, defined separately.

    Parameters
    ----------
    pd : PolyData or CellData, Optional
        A PolyData or a CellData instance. Dafault is None.

    cd : CellData, Optional
        A CellData instance, if the first argument is provided. Dafault is None.

    coords : ndarray, Optional.
        2d numpy array of floats, describing a pointcloud. Default is None.

    topo : ndarray, Optional.
        2d numpy array of integers, describing the topology of a polygonal mesh. 
        Default is None.

    celltype : int, Optional.
        An integer spcifying a valid celltype.

    Examples
    --------
    >>> from ..mesh import PolyData
    >>> from ..mesh.rgrid import grid
    >>> size = Lx, Ly, Lz = 100, 100, 100
    >>> shape = nx, ny, nz = 10, 10, 10
    >>> coords, topo = grid(size=size, shape=shape, eshape='H27')
    >>> pd = PolyData(coords=coords)
    >>> pd['A']['Part1'] = PolyData(topo=topo[:10])
    >>> pd['B']['Part2'] = PolyData(topo=topo[10:-10])
    >>> pd['C']['Part3'] = PolyData(topo=topo[-10:])

    See also
    --------
    :class:`..mesh.PolyData`
    :class:`..mesh.tri.trimesh.TriMesh`
    :class:`..mesh.pointdata.PointData`
    :class:`..mesh.celldata.CellData`

    """

    _point_array_class_ = PointCloud
    _point_class_ = PointData
    _frame_class_ = CartesianFrame
    _cell_class = NoneType
    _cell_classes_ = {
        8: Hexahedron,
        6: Wedge,
        4: Quadrilateral,
        9: Q9,
        10: TET10,
        3: Triangle,
        27: TriquadraticHexaHedron,
    }
    _passive_opacity_ = 0.3
    _active_opacity_ = 1.0
    _pv_config_key_ = ('pv', 'default')
    _k3d_config_key_ = ('k3d', 'default')

    def __init__(self, pd=None, cd=None, *args, coords=None, topo=None,
                 celltype=None, frame: FrameLike = None, newaxis: int = 2,
                 cell_fields=None, point_fields=None, parent: 'PolyData' = None,
                 **kwargs):
        self._reset_point_data()
        self._reset_cell_data()
        self._frame = frame
        self._newaxis = newaxis
        self._parent = parent
        self._config = DeepDict()
        self._init_config_()

        self.point_index_manager = IndexManager()
        self.cell_index_manager = IndexManager()

        if isinstance(pd, PointData):
            self.pointdata = pd
            if isinstance(cd, CellData):
                self.celldata = cd
        elif isinstance(pd, CellData):
            self.celldata = pd
            if isinstance(cd, PointData):
                self.pointdata = cd
        elif isinstance(cd, CellData):
            self.celldata = cd

        pkeys = self.__class__._point_class_._attr_map_
        ckeys = CellData._attr_map_

        if self.pointdata is not None:
            N = len(self.pointdata)
            GIDs = self.root().pim.generate_np(N)
            self.pd[pkeys['id']] = GIDs

        if self.celldata is not None:
            N = len(self.celldata)
            GIDs = self.root().cim.generate_np(N)
            self.cd[pkeys['id']] = GIDs
            try:
                pd = self.source().pd
            except Exception:
                pd = None
            self.cd.pd = pd
            self.cd.container = self

        super().__init__(*args, **kwargs)

        if self.pointdata is None and coords is not None:
            point_fields = {} if point_fields is None else point_fields
            pointtype = self.__class__._point_class_
            GIDs = self.root().pim.generate_np(coords.shape[0])
            point_fields[pkeys['id']] = GIDs
            self.pointdata = pointtype(coords=coords, frame=frame,
                                       newaxis=newaxis, stateful=True,
                                       fields=point_fields)

        if self.celldata is None and topo is not None:
            cell_fields = {} if cell_fields is None else cell_fields
            if celltype is None:
                celltype = self.__class__._cell_classes_.get(
                    topo.shape[1], None)
            elif isinstance(celltype, int):
                raise NotImplementedError
            if not issubclass(celltype, CellData):
                raise TypeError("Invalid cell type <{}>".format(celltype))

            if isinstance(topo, np.ndarray):
                topo = topo.astype(int)
            else:
                raise TypeError("Topo must be an 1d array of integers.")

            GIDs = self.root().cim.generate_np(topo.shape[0])
            cell_fields[ckeys['id']] = GIDs
            try:
                pd = self.source().pointdata
            except Exception:
                pd = None
            self.celldata = celltype(topo, fields=cell_fields, pointdata=pd)

        if self.celldata is not None:
            self.celltype = self.celldata.__class__
            self.celldata.container = self

    def __copy__(self, memo=None):
        cls = self.__class__
        result = cls(frame=self.frame)
        cfoo = copy if not memo is None else deepcopy
        if memo is not None:
            memo[id(self)] = result
        # self
        pointcls = cls._point_class_
        cellcls = self.celltype
        framecls = self._frame_class_
        if self.pointdata is not None:
            f = self.frame
            ax = cfoo(f.axes)
            if memo is not None:
                memo[id(f.axes)] = ax
            frame = framecls(ax, f.parent, order=f.order)
            db = cfoo(self.pd.db)
            if memo is not None:
                memo[id(self.pd.db)] = db
            result.pointdata = pointcls(
                frame=frame,
                db=db
            )
        if self.celldata is not None:
            pd = self.source()
            assert pd is not None
            db = cfoo(self.cd.db)
            if memo is not None:
                memo[id(self.cd.db)] = db
            result.celldata = cellcls(
                pointdata=pd,
                db=db
            )
        for k, v in self.items():
            if not isinstance(v, PolyData):
                v_ = cfoo(v)
                if memo is not None:
                    memo[id(v)] = v_
                result[k] = v

        # children
        l0 = len(self.address)
        for b in self.blocks(inclusive=False):
            pd, cd = None, None
            addr = b.address
            if len(addr) > l0:
                # pointdata
                if b.pointdata is not None:
                    f = b.frame
                    ax = cfoo(f.axes)
                    if memo is not None:
                        memo[id(f.axes)] = ax
                    frame = framecls(ax, f.parent, order=f.order)
                    db = cfoo(b.pd.db)
                    if memo is not None:
                        memo[id(b.pd.db)] = db
                    pd = pointcls(
                        frame=frame,
                        db=db
                    )
                # celldata
                if b.celldata is not None:
                    cellcls = b.celltype
                    db = cfoo(b.cd.db)
                    if memo is not None:
                        memo[id(b.cd.db)] = db
                    cd = cellcls(
                        pointdata=b.source(),
                        db=db
                    )
                    ct = b.celltype
                result[addr[l0:]] = PolyData(pd, cd, celltype=ct)
                # other data
                for k, v in b.items():
                    if not isinstance(v, PolyData):
                        v_ = cfoo(v)
                        if memo is not None:
                            memo[id(v)] = v_
                        b[k] = v

        result.__dict__.update(self.__dict__)
        return result

    @property
    def pd(self) -> PointData:
        return self.pointdata

    @property
    def cd(self) -> PolyCell:
        return self.celldata

    def simplify(self, inplace=True):
        # generalization of triangulation
        # cells should be broken into simplest representations
        raise NotImplementedError

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    @classmethod
    def read(cls, *args, **kwargs) -> 'PolyData':
        try:
            return cls.from_pv(pv.read(*args, **kwargs))
        except Exception:
            raise ImportError("Can't import file.")

    @classmethod
    def from_pv(cls, pvobj: pyVistaLike) -> 'PolyData':
        """
        Returns a :class:`..mesh.PolyData` instance from 
        a `pyvista.PolyData` or a `pyvista.UnstructuredGrid` instance.

        """
        celltypes = cls._cell_classes_.values()
        vtk_to_celltype = {v.vtkCellType: v for v in celltypes}

        if isinstance(pvobj, pv.PolyData):
            coords, topo = PolyData_to_mesh(pvobj)
            if isinstance(topo, dict):
                cells_dict = topo
            elif isinstance(topo, ndarray):
                ct = cls._cell_classes_[topo.shape[-1]]
                cells_dict = {ct.vtkCellType: topo}
        elif isinstance(pvobj, pv.UnstructuredGrid):
            ugrid = pvobj.cast_to_unstructured_grid()
            coords = pvobj.points.astype(float)
            cells_dict = ugrid.cells_dict
        else:
            try:
                ugrid = pvobj.cast_to_unstructured_grid()
                return cls.from_pv(ugrid)
            except Exception:
                raise TypeError

        A = CartesianFrame(dim=3)
        pd = PolyData(coords=coords, frame=A)  # this fails without a frame

        for vtkid, vtktopo in cells_dict.items():
            if vtkid in vtk_to_celltype:
                if vtkid == 5:
                    pass  # 3-noded triangles
                if vtkid == 10:
                    pass  # 10-noded tetrahedrons
                ct = vtk_to_celltype[vtkid]
                pd[vtkid] = PolyData(topo=vtktopo, celltype=ct, frame=A)
            else:
                msg = "The element type with vtkId <{}> is not jet supported here."
                raise NotImplementedError(msg.format(vtkid))

        return pd

    @property
    def config(self) -> DeepDict:
        """
        Returns the configuration object.

        Returns
        -------
        DeepDict
            The configuration object.

        Examples
        --------
        Assume `mesh` is a proper `PolyData` instance. Then to
        set configuration values related to plotting with `pyVista`,
        do the following:

        >>> mesh.config['pyvista', 'plot', 'color'] = 'red'
        >>> mesh.config['pyvista', 'plot', 'style'] = 'wireframe'

        Then, when it comes to plotting, you can specify your configuration
        with the `config_key` keyword argument:

        >>> mesh.pvplot(config_key=('pyvista', 'plot'))

        This way, you can store several different configurations for different
        scenarios.

        """
        return self._config

    def _init_config_(self):
        key = self.__class__._pv_config_key_
        self.config[key]['show_edges'] = True

    @property
    def pim(self) -> 'IndexManager':
        return self.point_index_manager

    @property
    def cim(self) -> 'IndexManager':
        return self.cell_index_manager

    @property
    def parent(self) -> 'PolyData':
        """Returns the parent of the object."""
        return self._parent

    @parent.setter
    def parent(self, value: 'PolyData'):
        """Sets the parent."""
        self._parent = value

    def is_source(self, key) -> bool:
        """
        Returns `True`, if the object is a valid source of data specified by `key`.

        Parameters
        ----------
        key : str
            A valid key to an `awkward` record.

        """
        key = 'x' if key is None else key
        return self.pointdata is not None and key in self.pointdata.fields

    def source(self, key=None) -> 'PolyData':
        """
        Returns the closest (going upwards in the hierarchy) block that holds on to data.
        If called without arguments, it is looking for a block with a valid pointcloud,
        definition, otherwise the field specified by the argument `key`.

        Parameters
        ----------
        key : str
            A valid key in any of the blocks with data. Default is None.

        See Also
        --------
        :class:`..mesh.PolyData`

        """
        key = 'x' if key is None else key
        if self.pointdata is not None:
            if key in self.pointdata.fields:
                return self
        if not self.is_root():
            return self.parent.source(key=key)
        else:
            raise KeyError("No data found with key '{}'".format(key))

    def blocks(self, *args, inclusive=False, blocktype=None, deep=True,
               **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over inner dictionaries.
        """
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def pointblocks(self, *args, **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over blocks with point data.
        """
        return filter(lambda i: i.pointdata is not None, self.blocks(*args, **kwargs))

    def cellblocks(self, *args, **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over blocks with cell data.
        """
        return filter(lambda i: i.celldata is not None, self.blocks(*args, **kwargs))

    @property
    def point_fields(self):
        """
        Returns the fields of all the pointdata of the object.

        Returns
        -------
        numpy.ndaray
            NumPy array of data keys.

        """
        pointblocks = list(self.pointblocks())
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def cell_fields(self):
        """
        Returns the fields of all the celldata of the object.

        Returns
        -------
        numpy.ndaray
            NumPy array of data keys.

        """
        cellblocks = list(self.cellblocks())
        m = map(lambda cb: cb.celldata.fields, cellblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def frame(self) -> FrameLike:
        """Returns the frame of the underlying pointcloud."""
        if self._frame is not None:
            return self._frame
        else:
            if self.is_source('x'):
                return self.pointdata.frame
        return self.source().frame

    @property
    def frames(self) -> ndarray:
        """Returnes the frames of the cells."""
        if self.celldata is not None:
            return self.celldata.frames

    @frames.setter
    def frames(self, value: ndarray):
        """Sets the frames of the cells."""
        assert self.celldata is not None
        if isinstance(value, ndarray):
            self.celldata.frames = value
        else:
            raise TypeError(('Type {} is not a supported' +
                             ' type to specify frames.').format(type(value)))

    def _reset_point_data(self):
        self.pointdata = None
        self.cell_index_manager = None

    def _reset_cell_data(self):
        self.celldata = None
        self.celltype = None

    def rewire(self, deep=True, imap=None):
        """
        Rewires topology according to the index mapping of the source object.

        Parameters
        ----------
        deep : bool
            If `True`, the action propagates down.

        Notes
        -----
        Unless node numbering was modified, subsequent executions have no effect
        after once called.

        """
        if not deep:
            if self.cd is not None:
                if imap is not None:
                    self.cd.rewire(imap=imap)
                else:
                    imap = self.source().pointdata.id
                    self.cd.rewire(imap=imap, invert=True)
        else:
            if imap is not None:
                [cb.rewire(imap=imap, deep=False) for
                 cb in self.cellblocks(inclusive=True)]
            else:
                [cb.rewire(deep=False) for
                 cb in self.cellblocks(inclusive=True)]
        return self

    def to_standard_form(self, inplace=True):
        """Transforms the problem to standard form."""
        if not self.is_root():
            raise NotImplementedError

        if not inplace:
            return deepcopy(self).to_standard_form(inplace=True)

        # merge points and point related data
        # + decorate the points with globally unique ids
        im = IndexManager()
        pointtype = self.__class__._point_class_
        pointblocks = list(self.pointblocks(inclusive=True))
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        fields = np.unique(np.array(list(m)).flatten())
        m = map(lambda pb: pb.pointdata.x, pointblocks)
        X, frame, axis = np.vstack(list(m)), self._frame, self._newaxis
        if len(fields) > 0:
            point_fields = {}
            data = {f: [] for f in fields}
            for pb in pointblocks:
                GIDs = im.generate_np(len(pb.pointdata))
                pb.pointdata['id'] = GIDs
                for f in fields:
                    if f in pb.pointdata.fields:
                        data[f].append(pb.pointdata[f].to_numpy())
                    else:
                        data[f].append(np.zeros(len(pb.pointdata)))
            data.pop('x', None)
            for f in data.keys():
                nd = np.max([len(d.shape) for d in data[f]])
                fdata = list(map(lambda arr: atleastnd(arr, nd), data[f]))
                point_fields[f] = np.concatenate(fdata, axis=0)
        else:
            point_fields = None
        self.pointdata = pointtype(coords=X, frame=frame, newaxis=axis,
                                   stateful=True, fields=point_fields)

        # merge cells and cell related data
        # + rewire the topology based on the ids set in the previous block
        cellblocks = list(self.cellblocks(inclusive=True))
        m = map(lambda pb: pb.celldata.fields, cellblocks)
        fields = np.unique(np.array(list(m)).flatten())
        if len(fields) > 0:
            ndim = {f: [] for f in fields}
            for cb in cellblocks:
                imap = cb.source().pointdata.id
                # cb.celldata.rewire(imap=imap)  # this has been done at joining parent
                for f in fields:
                    if f in cb.celldata.fields:
                        ndim[f].append(len(cb.celldata[f].to_numpy().shape))
                    else:
                        cb.celldata[f] = np.zeros(len(cb.celldata))
                        ndim[f].append(1)
            ndim = {f: np.max(v) for f, v in ndim.items()}
            for cb in cellblocks:
                cb.celldata[f] = atleastnd(cb.celldata[f].to_numpy(), ndim[f])

        # free resources
        for pb in self.pointblocks(inclusive=False):
            pb._reset_point_data()

        return self

    def points(self, *args, return_inds=False, from_cells=False, **kwargs) -> PointCloud:
        """
        Returns the points as a :class:`..mesh.space.PointCloud` instance.

        Notes
        -----
        Opposed to :func:`coords`, which returns the coordiantes, it returns the points 
        of a mesh as vectors.

        See Also
        --------
        :class:`..mesh.space.PointCloud`
        :func:`coords`

        """
        target = self.frame
        if from_cells:
            inds_ = np.unique(self.topology())
            x, inds = self.root().points(from_cells=False, return_inds=True)
            imap = inds_to_invmap_as_dict(inds)
            inds = remap_topo_1d(inds_, imap)
            coords, inds = x[inds, :], inds_
        else:
            __cls__ = self.__class__._point_array_class_
            coords, inds = [], []
            for pb in self.pointblocks(inclusive=True):
                x = pb.pd.x
                fr = pb.frame
                i = pb.pd.id
                v = Vector(x, frame=fr)
                coords.append(v.show(target))
                inds.append(i)
            coords = np.vstack(list(coords))
            inds = np.concatenate(inds).astype(int)
        __cls__ = self.__class__._point_array_class_
        points = __cls__(coords, frame=target, inds=inds)
        if return_inds:
            return points, inds
        return points

    def coords(self, *args, return_inds=False, from_cells=False, **kwargs) -> VectorBase:
        """Returns the coordinates as an array."""
        if return_inds:
            p, inds = self.points(return_inds=True, from_cells=from_cells)
            return p.show(*args, **kwargs), inds
        else:
            return self.points(from_cells=from_cells).show(*args, **kwargs)

    def surface(self):
        assert self.celldata is not None, "There are no cells here."
        assert self.celldata.NDIM == 3, "This is only for 3d cells."
        coords, topo = self.cd.extract_surface(detach=False)
        pointtype = self.__class__._point_class_
        frame = self.source().frame
        pd = pointtype(coords=coords, frame=frame)
        cd = Triangle(topo=topo, pointdata=pd)
        return self.__class__(pd, cd, frame=frame)

    def cells(self):
        """
        This should be the same to topology, what point is to coords,
        with no need to copy the underlying mechanism.

        The relationship of resulting object to the topology of a mesh should 
        be similar to that of `PointCloud` and the points in 3d space.

        """
        pass

    def topology(self, *args, return_inds=False, triangulate=True, **kwargs):
        """
        Returns the topology as either a `numpy` or an `awkward` array.

        Notes
        -----
        The call automatically propagates down.

        """
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.topology(), blocks))
        widths = np.concatenate(list(map(lambda t: t.widths(), topo)))
        jagged = not np.all(widths == widths[0])
        if jagged:
            if return_inds:
                raise NotImplementedError
            return TopologyArray(*topo)
        else:
            topo = np.vstack([t.to_numpy() for t in topo])
            if return_inds:
                inds = list(map(lambda i: i.celldata.id, blocks))
                return topo, np.concatenate(inds)
            else:
                return topo

    def detach(self, nummrg=False) -> 'PolyData':
        pd = PolyData(self.root().pd, frame=self.frame)
        l0 = len(self.address)
        if self.celldata is not None:
            db = deepcopy(self.cd.db)
            cd = self.celltype(pointdata=pd, db=db)
            pd.celldata = cd
            pd.celltype = self.celltype
        for cb in self.cellblocks(inclusive=False):
            addr = cb.address
            if len(addr) > l0:
                db = deepcopy(cb.cd.db)
                cd = cb.celltype(pointdata=pd, db=db)
                pd[addr[l0:]] = PolyData(None, cd)
                assert pd[addr[l0:]].celldata is not None
        if nummrg:
            pd.nummrg()
        return pd

    def nummrg(self, store_indices=True):
        if not self.is_root():
            self.root().nummrg()
            return self
        topo = self.topology()
        inds = np.unique(topo)
        pointtype = self.__class__._point_class_
        self.pointdata = pointtype(db=self.pd[inds])
        if store_indices:
            self.pointdata._wrapped['gid'] = self.pd.id
        imap = inds_to_invmap_as_dict(self.pd.id)
        [cb.rewire(imap=imap) for cb in self.cellblocks(inclusive=True)]
        self.pointdata._wrapped['id'] = np.arange(len(self.pd))
        return self

    def move(self, v: VectorLike, frame: FrameLike = None):
        """Moves the object."""
        if self.is_root():
            pc = self.points()
            pc.move(v, frame)
            self.pointdata['x'] = pc.array
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            pc.move(v, frame)
            root.pointdata['x'] = pc.array
        return self

    def rotate(self, *args, **kwargs):
        """Rotates the object."""
        if self.is_root():
            pc = self.points()
            pc.rotate(*args, **kwargs)
            self.pointdata['x'] = pc.show(self.frame)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            pc.rotate(*args, **kwargs)
            root.pointdata['x'] = pc.show(self.frame)
        return self

    def cells_at_nodes(self, *args, **kwargs):
        """Returns the neighbouring cells of nodes."""
        topo = self.topology()
        return cells_at_nodes(topo, *args, **kwargs)

    def cells_around_cells(self, radius=None, frmt='dict'):
        """Returns the neares cells to cells."""
        if radius is None:
            # topology based
            raise NotImplementedError
        else:
            return cells_around(self.centers(), radius, frmt=frmt)

    def nodal_adjacency_matrix(self, *args, **kwargs):
        """
        Returns the nodal adjecency matrix. The arguments are 
        forwarded to the corresponding utility function (see below)
        alongside the topology of the mesh as the first argument.

        See also
        --------
        :func:`..mesh.topo.topo.nodal_adjacency`

        """
        topo = self.topology()
        return nodal_adjacency(topo, *args, **kwargs)

    def number_of_cells(self) -> int:
        """Returns the number of cells."""
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self) -> int:
        """Returns the number of points"""
        return len(self.root().pointdata)

    def cells_coords(self, *args, _topo=None, **kwargs) -> ndarray:
        """Returns the coordiantes of the cells in extrensic format."""
        _topo = self.topology() if _topo is None else _topo
        return cells_coords(self.root().coords(), _topo)

    def center(self, target: FrameLike = None) -> ndarray:
        """Returns the center of the pointcloud of the mesh."""
        if self.is_root():
            return self.points().center(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            return pc.center(target)

    def centers(self, *args, target: FrameLike = None, **kwargs) -> ndarray:
        """Returns the centers of the cells."""
        if self.is_root():
            coords = self.points().show(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            coords = pc.show(target)
        return cell_center_bulk(coords, self.topology(*args, **kwargs))

    def centralize(self, target: FrameLike = None) -> 'PolyData':
        """
        Centralizes the coordinats of the pointcloud of the mesh
        and returns the object for continuation.
        """
        pc = self.root().points()
        pc.centralize(target)
        self.pointdata['x'] = pc.show(self.frame)
        return self

    def k_nearest_cell_neighbours(self, k, *args, knn_options=None, **kwargs):
        """
        Returns the k closest neighbours of the cells of the mesh, based
        on the centers of each cell.

        The argument `knn_options` is passed to the KNN search algorithm,
        the rest to the `centers` function of the mesh.
        
        Examples
        --------
        >>> from sigmaepsilon.mesh.grid import Grid
        >>> from sigmaepsilon.mesh import KNN
        >>> size = 80, 60, 20
        >>> shape = 10, 8, 4
        >>> grid = Grid(size=size, shape=shape, eshape='H8')
        >>> X = grid.centers()
        >>> i = KNN(X, X, k=3, max_distance=10.0)
        """
        c = self.centers(*args, **kwargs)
        knn_options = {} if knn_options is None else knn_options
        return KNN(c, c, k=k, **knn_options)

    def areas(self, *args, **kwargs) -> ndarray:
        """Returns the areas."""
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        blocks2d = filter(lambda b: b.celltype.NDIM < 3, blocks)
        amap = map(lambda b: b.celldata.areas(coords=coords), blocks2d)
        return np.concatenate(list(amap))

    def area(self, *args, **kwargs) -> float:
        """Returns the sum of areas in the model."""
        return np.sum(self.areas(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """Returns the volumes of the cells."""
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.volumes(coords=coords), blocks)
        return np.concatenate(list(vmap))

    def volume(self, *args, **kwargs) -> float:
        """Returns the net volume of the mesh."""
        return np.sum(self.volumes(*args, **kwargs))

    def index_of_closest_point(self, target, *args, **kwargs) -> int:
        """Returns the index of the closest point to a target."""
        return index_of_closest_point(self.coords(), target)

    def index_of_closest_cell(self, target, *args, **kwargs) -> int:
        """Returns the index of the closest cell to a target."""
        return index_of_closest_point(self.centers(), target)

    def set_nodal_distribution_factors(self, *args, **kwargs):
        self.nodal_distribution_factors(*args, store=True, **kwargs)

    def nodal_distribution_factors(self, *args, assume_regular=False,
                                   key='ndf', store=False, measure='volume',
                                   load=None, weights=None, **kwargs) -> ndarray:
        if load is not None:
            if isinstance(load, str):
                blocks = self.cellblocks(inclusive=True)
                def foo(b): return b.celldata._wrapped[load].to_numpy()
                return np.vstack(list(map(foo, blocks)))

        topo, inds = self.topology(return_inds=True)

        if measure == 'volume':
            weights = self.volumes()
        elif measure == 'uniform':
            weights = np.ones(topo.shape[0], dtype=float)

        argsort = np.argsort(inds)
        topo = topo[argsort]
        weights = weights[argsort]
        if not assume_regular:
            topo, _ = regularize(topo)
        factors = nodal_distribution_factors(topo, weights)
        if store:
            blocks = self.cellblocks(inclusive=True)

            def foo(b): return b.celldata.set_nodal_distribution_factors(
                factors, key=key)
            list(map(foo, blocks))
        return factors

    def to_vtk(self, *args, deepcopy=True, fuse=True, deep=True,
               scalars=None, detach=True, **kwargs):
        """
        Returns the mesh as a `vtk` oject, and optionally fetches
        data.

        """
        if not __hasvtk__:
            raise ImportError
        coords = self.root().coords()
        blocks = list(self.cellblocks(inclusive=True, deep=deep))
        def mesh(c, t): return detach_mesh_bulk(c, t) if detach else (c, t)
        if fuse:
            if len(blocks) == 1:
                topo = blocks[0].celldata.nodes.astype(np.int64)
                ugrid = mesh_to_vtk(*mesh(coords, topo),
                                    blocks[0].celltype.vtkCellType, deepcopy)
                return ugrid
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(blocks))
            for i, block in enumerate(blocks):
                topo = block.celldata.nodes.astype(np.int64)
                ugrid = mesh_to_vtk(*mesh(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                mb.SetBlock(i, ugrid)
            return mb
        else:
            needsdata = isinstance(scalars, str)
            res, plotdata = [], []
            for i, block in enumerate(blocks):
                if needsdata:
                    pdata = None
                    if self.pointdata is not None:
                        if scalars in self.pointdata.fields:
                            pdata = self.pointdata[scalars].to_numpy()
                    if pdata is None and scalars in self.celldata.fields:
                        pdata = self.celldata[scalars].to_numpy()
                    plotdata.append(pdata)
                # the next line handles regular topologies only
                topo = block.celldata.topology().to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*mesh(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                res.append(ugrid)

            if needsdata:
                return res, plotdata
            else:
                return res

    def to_pv(self, *args, fuse=True, deep=True, scalars=None, **kwargs):
        """
        Returns the mesh as a `pyVista` oject, optionally set up with data.

        """
        if not __haspyvista__:
            raise ImportError
        data = None
        if isinstance(scalars, str) and not fuse:
            vtkobj, data = self.to_vtk(*args, fuse=False, deep=deep,
                                       scalars=scalars, **kwargs)
        else:
            vtkobj = self.to_vtk(*args, fuse=fuse, deep=deep, **kwargs)
            data = None
        if fuse:
            assert data is None
            multiblock = pv.wrap(vtkobj)
            try:
                multiblock.wrap_nested()
            except AttributeError:
                pass
            return multiblock
        else:
            if data is None:
                return [pv.wrap(i) for i in vtkobj]
            else:
                res = []
                for ugrid, d in zip(vtkobj, data):
                    pvobj = pv.wrap(ugrid)
                    if isinstance(d, ndarray):
                        pvobj[scalars] = d
                    res.append(pvobj)
                return res

    @suppress
    def to_k3d(self, *args, scene=None, deep=True, menu_visibility=True, scalars=None,
               config_key=None, color_map=None, detach=False, show_edges=True, **kwargs):
        assert __hask3d__, "The python package 'k3d' must be installed for this"
        if scene is None:
            scene = k3d.plot(menu_visibility=menu_visibility)
        vertices = self.root().coords().astype(np.float32)

        pvparams = dict(wireframe=False)
        if config_key is None:
            config_key = self.__class__._k3d_config_key_

        for b in self.cellblocks(inclusive=True, deep=deep):
            params = copy(pvparams)
            params.update(b.config[config_key])
            if 'color' in params:
                if isinstance(params['color'], str):
                    hexstr = mpl.colors.to_hex(params['color'])
                    params['color'] = int("0x" + hexstr[1:], 16)
            if color_map is not None:
                params['color_map'] = color_map
            if b.celltype.NDIM == 1:
                i = b.cd.topology().to_numpy()
                scene += k3d.lines(vertices, i.astype(np.uint32),
                                   indices_type='segment', **params)
            elif b.celltype.NDIM == 2:
                i = b.cd.to_triangles()
                if 'side' in params:
                    if params['side'].lower() == 'both':
                        params['side'] = 'front'
                        scene += k3d.mesh(vertices,
                                          i.astype(np.uint32), **params)
                        params['side'] = 'back'
                        scene += k3d.mesh(vertices,
                                          i.astype(np.uint32), **params)
                    else:
                        scene += k3d.mesh(vertices,
                                          i.astype(np.uint32), **params)
                else:
                    scene += k3d.mesh(vertices, i.astype(np.uint32), **params)
                if show_edges:
                    scene += k3d.mesh(vertices, i.astype(np.uint32),
                                      wireframe=True, color=0)
            elif b.celltype.NDIM == 3:
                i = b.surface().topology()
                #i = b.surface(detach=False, triangulate=True).topology()
                scene += k3d.mesh(vertices, i.astype(np.uint32), **params)
                if show_edges:
                    scene += k3d.mesh(vertices, i.astype(np.uint32),
                                      wireframe=True, color=0)
        return scene

    def plot(self, *args, notebook=False, backend=None, config_key=None, **kwargs):
        if notebook and backend == 'k3d':
            return self.k3dplot(*args, config_key=config_key, **kwargs)
        return self.pvplot(*args, notebook=notebook, config_key=config_key, **kwargs)

    def k3dplot(self, scene=None, *args, menu_visibility=True, **kwargs):
        if scene is None:
            scene = k3d.plot(menu_visibility=menu_visibility)
        return self.to_k3d(*args, scene=scene, **kwargs)

    def pvplot(self, *args, deepcopy=True, jupyter_backend='pythreejs',
               show_edges=True, notebook=False, theme='document',
               scalars=None, window_size=None, return_plotter=False,
               config_key=None, plotter=None, cmap=None, camera_position=None,
               lighting=False, edge_color=None, **kwargs):
        if not __haspyvista__:
            raise ImportError('You need to install `pyVista` for this.')
        if scalars is None:
            polys = self.to_pv(deepcopy=deepcopy, fuse=False)
        else:
            polys = self.to_pv(deepcopy=deepcopy, scalars=scalars, fuse=False)

        if isinstance(theme, str):
            try:
                pv.set_plot_theme(theme)
            except Exception:
                if theme == 'dark':
                    theme = themes.DarkTheme()
                    theme.lighting = False
                    theme.show_edges = True
                elif theme == 'bw':
                    theme.color = 'black'
                    theme.lighting = True
                    theme.show_edges = True
                    theme.edge_color = 'white'
                    theme.background = 'white'
            theme = pv.global_theme

        if lighting is not None:
            theme.lighting = lighting
        if show_edges is not None:
            theme.show_edges = show_edges
        if edge_color is not None:
            theme.edge_color = edge_color

        if plotter is None:
            pvparams = dict()
            if window_size is not None:
                pvparams.update(window_size=window_size)
            pvparams.update(kwargs)
            pvparams.update(notebook=notebook)
            pvparams.update(theme=theme)
            plotter = pv.Plotter(**pvparams)
        else:
            return_plotter = True

        if camera_position is not None:
            plotter.camera_position = camera_position

        pvparams = dict(show_edges=show_edges)
        blocks = self.cellblocks(inclusive=True, deep=True)
        if config_key is None:
            config_key = self.__class__._pv_config_key_
        for block, poly in zip(blocks, polys):
            params = copy(pvparams)
            params.update(block.config[config_key])
            if cmap is not None:
                params['cmap'] = cmap
            plotter.add_mesh(poly, **params)
        if return_plotter:
            return plotter
        if notebook:
            return plotter.show(jupyter_backend=jupyter_backend)
        else:
            return plotter.show()

    def __join_parent__(self, parent: DeepDict, key: Hashable = None):
        super().__join_parent__(parent, key)
        if self.pointdata is not None:
            GIDs = self.root().pim.generate_np(len(self.pointdata))
            self.pointdata['id'] = GIDs
        if self.celldata is not None:
            GIDs = self.root().cim.generate_np(len(self.celldata))
            self.celldata['id'] = GIDs
            if self.celldata.pd is None:
                self.celldata.pd = self.source().pd
            self.celldata.container = self
        self.rewire(deep=True)

    def __leave_parent__(self):
        if self.pointdata is not None:
            self.root().pim.recycle(self.poitdata.id)
            dbkey = self.pointdata.__class__._attr_map_['id']
            del self.pointdata._wrapped[dbkey]
        if self.celldata is not None:
            self.root().cim.recycle(self.celldata.id)
            dbkey = self.celldata.__class__._attr_map_['id']
            del self.celldata._wrapped[dbkey]
        super().__leave_parent__()

    def __repr__(self):
        return 'PolyData(%s)' % (dict.__repr__(self))


class IndexManager(object):
    """This object ought to guarantee, that every cell in a 
    model has a unique ID."""

    def __init__(self, start=0):
        self.queue = []
        self.next = start

    def generate_np(self, n=1):
        if n == 1:
            return self.generate(1)
        else:
            return np.array(self.generate(n))

    def generate(self, n=1):
        nQ = len(self.queue)
        if nQ > 0:
            if n == 1:
                res = self.queue.pop()
            else:
                if nQ >= n:
                    res = self.queue[:n]
                    del self.queue[:n]
                else:
                    res = copy(self.queue)
                    res.extend(range(self.next, self.next + n - nQ))
                    self.queue = []
                self.next += n - nQ
        else:
            if n == 1:
                res = self.next
            else:
                res = list(range(self.next, self.next + n))
            self.next += n
        return res

    def recycle(self, *args, **kwargs):
        for a in args:
            if isinstance(a, Iterable):
                self.queue.extend(a)
            else:
                self.queue.append(a)
