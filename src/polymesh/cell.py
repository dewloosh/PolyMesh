from typing import Union, MutableMapping, Iterable
from typing import Tuple, List, Callable

import numpy as np
from numpy import ndarray
from sympy import Matrix, lambdify

from neumann import atleast1d, atleast2d, ascont
from neumann.utils import to_range_1d

from .celldata import CellData
from .utils.utils import (
    jacobian_matrix_bulk,
    points_of_cells,
    pcoords_to_coords,
    pcoords_to_coords_1d,
    cells_coords,
    lengths_of_lines,
)
from .utils.tri import area_tri_bulk
from .utils.tet import tet_vol_bulk
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from .utils.topology.topo import detach_mesh_bulk, rewire
from .utils.topology import transform_topo
from .utils.tri import triangulate_cell_coords
from .utils import cell_center, cell_center_2d
from .topoarray import TopologyArray
from .space import CartesianFrame
from .triang import triangulate
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
        return cell_center(cls.lcoords())

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
    def generate_class_functions(
        cls, return_symbolic: bool = True, update: bool = True
    ) -> Tuple:
        """
        Generates functions to evaulate shape functions, their derivatives
        and the shape function matrices using SymPy. For this to work, the
        'polybase' and 'lcoords' class methods must be implemented.

        Parameters
        ----------
        return_symbolic : bool, Optional
            If True, the function returns symbolic expressions of shape functions
            and their derivatives. Default is True.
        update : bool, Optional
            If True, class methods are updated with the generated versions.
            Default is True.
        """
        nN = cls.NNODE
        nD = cls.NDIM
        nDOF = getattr(cls, "NDOFN", 3)
        locvars, monoms = cls.polybase()
        monoms.pop(0)
        lcoords = cls.lcoords()
        if nD == 1:
            lcoords = np.reshape(lcoords, (nN, 1))

        def subs(lpos):
            return {v: lpos[i] for i, v in enumerate(locvars)}

        def mval(lpos):
            return [m.evalf(subs=subs(lpos)) for m in monoms]

        M = np.ones((nN, nN), dtype=float)
        M[:, 1:] = np.vstack([mval(loc) for loc in lcoords])
        coeffs = np.linalg.inv(M)
        monoms.insert(0, 1)
        shp = Matrix([np.dot(coeffs[:, i], monoms) for i in range(nN)])
        dshp = Matrix([[f.diff(m) for m in locvars] for f in shp])
        _shpf = lambdify([locvars], shp[:, 0].T, "numpy")
        _dshpf = lambdify([locvars], dshp, "numpy")

        def shpf(p: ndarray) -> ndarray:
            """
            Evaluates the shape functions at multiple points in the
            master domain.
            """
            r = np.stack([_shpf(p[i])[0] for i in range(len(p))])
            return ascont(r)

        def shpmf(p: ndarray, ndof: int = nDOF) -> ndarray:
            """
            Evaluates the shape function matrix at multiple points
            in the master domain.
            """
            nP = p.shape[0]
            eye = np.eye(ndof, dtype=float)
            shp = shpf(p)
            res = np.zeros((nP, ndof, nN * ndof), dtype=float)
            for iP in range(nP):
                for i in range(nN):
                    res[iP, :, i * ndof : (i + 1) * ndof] = eye * shp[iP, i]
            return ascont(res)

        def dshpf(p: ndarray) -> ndarray:
            """
            Evaluates the shape function derivatives at multiple points
            in the master domain.
            """
            r = np.stack([_dshpf(p[i]) for i in range(len(p))])
            return ascont(r)

        if update:
            cls.shpfnc = shpf
            cls.shpmfnc = shpmf
            cls.dshpfnc = dshpf

        if return_symbolic:
            return shp, dshp, shpf, shpmf, dshpf
        else:
            return shpf, shpmf, dshpf

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
            An array of shape (nP, nNE) where nP and nNE are the number of
            evaluation points and shape functions. If there is only one
            evaluation point, the returned array is one dimensional.
        """
        pcoords = np.array(pcoords)
        if cls.shpfnc is None:
            cls.generate_class_functions(update=True)
        if cls.NDIM == 3:
            if len(pcoords.shape) == 1:
                pcoords = atleast2d(pcoords, front=True)
                return cls.shpfnc(pcoords).astype(float)
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
        nDOFN = getattr(cls, "NDOFN", None)
        pcoords = np.array(pcoords)
        if cls.shpmfnc is None:
            cls.generate_class_functions(update=True)
        if cls.NDIM == 3:
            if len(pcoords.shape) == 1:
                pcoords = atleast2d(pcoords, front=True)
                if nDOFN:
                    return cls.shpmfnc(pcoords, nDOFN).astype(float)
                else:
                    return cls.shpmfnc(pcoords).astype(float)
        if nDOFN:
            return cls.shpmfnc(pcoords, nDOFN).astype(float)
        else:
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
        pcoords = np.array(pcoords)
        if cls.dshpfnc is None:
            cls.generate_class_functions(update=True)
        if cls.NDIM == 3:
            if len(pcoords.shape) == 1:
                pcoords = atleast2d(pcoords, front=True)
                return cls.dshpfnc(pcoords).astype(float)
        return cls.dshpfnc(pcoords).astype(float)

    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return measures for each cell in the database."""
        raise NotImplementedError

    def measure(self, *args, **kwargs) -> float:
        """Ought to return the net measure for the cells in the
        database as a group."""
        return np.sum(self.measures(*args, **kwargs))

    def area(self, *args, **kwargs) -> float:
        """Returns the total area of the cells in the database. Only for 2d entities."""
        return np.sum(self.areas(*args, **kwargs))

    def areas(self, *args, **kwargs) -> ndarray:
        """Ought to return the areas of the individuall cells in the database."""
        raise NotImplementedError

    def volume(self, *args, **kwargs) -> float:
        """Returns the volume of the cells in the database."""
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """Ought to return the volumes of the individual cells in the database."""
        raise NotImplementedError

    def extract_surface(self, detach:bool=False):
        """Extracts the surface of the mesh. Only for 3d meshes."""
        raise NotImplementedError

    def jacobian_matrix(self, *, dshp: ndarray = None, **__) -> ndarray:
        """
        Returns the jacobian matrices.

        Parameters
        ----------
        dshp : numpy.ndarray
            3d array of shape function derivatives for the master cell,
            evaluated at some points. The array must have a shape of
            (nG, nNE, nD), where nG, nNE and nD are he number of evaluation
            points, nodes per cell and spatial dimensions.

        Returns
        -------
        numpy.ndarray
            A 4d array of shape (nE, nG, nD, nD), where nE, nG and nD
            are the number of elements, evaluation points and spatial
            dimensions. The number of evaluation points in the output
            is governed by the parameter 'dshp'.
        """
        ecoords = self.local_coordinates()
        return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *, jac: ndarray = None, **kwargs) -> Union[float, ndarray]:
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

    def source_coords(self) -> ndarray:
        """
        Returns the coordinates of the hosting pointcloud.
        """
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        return coords

    def points_of_cells(
        self,
        *,
        points: Union[float, Iterable] = None,
        cells: Union[int, Iterable] = None,
        target: Union[str, CartesianFrame] = "global"
    ) -> ndarray:
        """
        Returns the points of the cells as a NumPy array.
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        if isinstance(target, str):
            assert target.lower() in ["global", "g"]
        else:
            raise NotImplementedError

        coords = self.source_coords()
        topo = self.topology().to_numpy()[cells]
        ecoords = points_of_cells(coords, topo, centralize=False)

        if points is None:
            return ecoords
        else:
            points = np.array(points)

        shp = self.shape_function_values(points)
        if len(shp) == 3:  # variable metric cells
            shp = shp if len(shp) == 2 else shp[cells]

        return pcoords_to_coords(points, ecoords, shp)  # (nE, nP, nD)

    def local_coordinates(self, *, target: CartesianFrame = None) -> ndarray:
        """
        Returns local coordinates of the cells as a 3d float
        numpy array.

        Parameters
        ----------
        target : CartesianFrame, Optional
            A target frame. If provided, coordinates are returned in
            this frame, otherwise they are returned in the local frames
            of the cells. Default is None.
        """
        if isinstance(target, CartesianFrame):
            frames = target.show()
        else:
            frames = self.frames
        topo = self.topology().to_numpy()
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        return points_of_cells(coords, topo, local_axes=frames, centralize=True)

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

    def rewire(self, imap: MapLike = None, invert: bool = False) -> "PolyCell":
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping of the j-th node
        of the i-th cell happens the following way:

        topology_new[i, j] = imap[topology_old[i, j]]

        The object is returned for continuation.

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
        return self
    
    def glob_to_loc(self, x: ndarray) -> ndarray:
        """
        Returns the local coordinates of the input points for each
        cell in the block. The input 'x' can describe a single (1d array), 
        or several vectors at once (2d array).
        
        Notes
        -----
        This function is useful when detecting if two bodies touch each other or not,
        and if they do, where.
        
        Parameters
        ----------
        x: numpy.ndarray
            A single point in 3d space as an 1d array, or a collection of points
            as a 2d array.
            
        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nE, nP, nD), where nP is the number of points in 'x',
            nE is the number of cells in the block and nD is the number of spatial dimensions.
        """
        pass
    
    def loc_to_glob(self, x: ndarray) -> ndarray:
        pass
    
    def pip(self, x: ndarray) -> ndarray:
        pass
    
    def to_simplices(self) -> Tuple[ndarray]:
        NDIM = self.__class__.NDIM
        if NDIM == 1:
            return self.to_simplices()
        elif NDIM == 2:
            return self.to_triangles()
        elif NDIM == 3:
            return self.to_tetrahedra()

    def _simplify(self) -> Tuple[ndarray]:
        # this should modify the object inplace
        raise NotImplementedError
        return self.to_simplices()
        
        
class PolyCell1d(PolyCell):
    """Base class for 1d cells"""

    NDIM = 1

    def lenth(self):
        """Returns the total length of the cells in
        the database."""
        return np.sum(self.lengths())

    def lengths(self) -> ndarray:
        """
        Returns the lengths as a NumPy array.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return lengths_of_lines(coords, topo)
    
    def area(self) -> ndarray:
        # should return area of the surface of the volume
        raise NotImplementedError

    def areas(self) -> ndarray:
        """
        Returns the areas as a NumPy array.
        """
        areakey = self._dbkey_areas_
        if areakey in self.fields:
            return self[areakey].to_numpy()
        else:
            return np.ones((len(self)))

    def volumes(self) -> ndarray:
        """
        Returns the volumes as a NumPy array.
        """
        return self.lengths() * self.areas()

    def measures(self) -> ndarray:
        return self.lengths()

    def points_of_cells(
        self,
        *,
        points: Union[float, Iterable] = None,
        cells: Union[int, Iterable] = None,
        flatten: bool = False,
        target: Union[str, CartesianFrame] = "global",
        rng: Iterable = None,
        **kwargs
    ) -> ndarray:
        if isinstance(target, str):
            assert target.lower() in ["global", "g"]
        else:
            raise NotImplementedError
        topo = kwargs.get("topo", self.topology().to_numpy())
        coords = kwargs.get("coords", None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        ecoords = points_of_cells(coords, topo, centralize=False)
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
            points = np.array(points)
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        points, rng = to_range_1d(points, source=rng, target=[0, 1]).flatten(), [0, 1]
        res = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)

        if not flatten:
            nE = ecoords.shape[0]
            nP = points.shape[0]
            res = res.reshape(nE, nP, res.shape[-1])  # (nE, nP, nD)

        return res


class PolyCell2d(PolyCell):
    """Base class for 2d cells"""

    NDIM = 2

    def area(self) -> float:
        """
        Returns the total area of the cells in the block.
        """
        return np.sum(self.areas())

    @classmethod
    def trimap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of T3 triangles.
        """
        _, t, _ = triangulate(points=cls.lcoords())
        return t

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Ought to return the local coordinates of the center of the
        master element.

        Returns
        -------
        numpy.ndarray
        """
        return cell_center_2d(cls.lcoords())

    def to_triangles(self) -> ndarray:
        """
        Returns the topology as a collection of T3 triangles.
        """
        t = self.topology().to_numpy()
        return transform_topo(t, self.trimap())

    def areas(self) -> ndarray:
        """
        Returns the areas of the cells.
        """
        nE = len(self)
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        frames = self.frames
        ec = points_of_cells(coords, topo, local_axes=frames)
        trimap = self.__class__.trimap()
        ec_tri = triangulate_cell_coords(ec, trimap)
        areas_tri = area_tri_bulk(ec_tri)
        res = np.sum(areas_tri.reshape(nE, int(len(areas_tri) / nE)), axis=1)
        return res

    def volumes(self) -> ndarray:
        """
        Returns the volumes of the cells.
        """
        areas = self.areas()
        t = self.thickness()
        return areas * t

    def measures(self) -> ndarray:
        """
        Returns the areas of the cells.
        """
        return self.areas()

    def local_coordinates(self, *_, target: CartesianFrame = None) -> ndarray:
        ec = super(PolyCell2d, self).local_coordinates(target=target)
        return ascont(ec[:, :, :2])

    def thickness(self) -> ndarray:
        """
        Returns the thicknesses of the cells. If not set, a thickness
        of 1.0 is returned for each cell.
        """
        dbkey = self._dbkey_thickness_
        if dbkey in self.fields:
            t = self.db[dbkey].to_numpy()
        else:
            t = np.ones(len(self), dtype=float)
        return t


class PolyCell3d(PolyCell):
    """Base class for 3d cells"""

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measures(self, *args, **kwargs):
        return self.volumes(*args, **kwargs)
    
    @classmethod
    def tetmap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of T3 triangles.
        """
        raise NotImplementedError

    def to_tetrahedra(self) -> np.ndarray:
        """
        Returns the topology as a collection of TET4 tetrahedra.
        """
        t = self.topology().to_numpy()
        return transform_topo(t, self.tetmap())

    def to_vtk(self, detach:bool=False):
        """
        Returns the block as a VTK object.
        """
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
            """
            Returns the block as a pyVista object.
            """
            return pv.wrap(self.to_vtk(detach=detach))

    def extract_surface(self, detach:bool=False) -> Tuple[ndarray]:
        """
        Extracts the surface of the object.
        """
        pvs = self.to_pv(detach=detach).extract_surface(pass_pointid=True)
        s = pvs.triangulate().cast_to_unstructured_grid()
        topo = s.cells_dict[5]
        imap = s.point_data["vtkOriginalPointIds"]
        topo = rewire(topo, imap)
        if detach:
            return s.points, topo
        else:
            return self.container.root().coords(), topo

    def boundary(self, detach=False) -> Tuple[ndarray]:
        """
        Returns the boundary of the block as 2 NumPy arrays.
        """
        return self.extract_surface(detach=detach)

    def volumes(self):
        # NOTE implement `tetmap` class attribute, then it can be used
        # to check if calculation should be based on splitting or Gauss integration
        # Look at the examples for Gauss integration at child classes.
        coords = self.container.root().coords()
        topo = self.topology().to_numpy()
        topo_tet = self.to_tetrahedra()
        volumes = tet_vol_bulk(cells_coords(coords, topo_tet))
        res = np.sum(
            volumes.reshape(topo.shape[0], int(len(volumes) / topo.shape[0])), axis=1
        )
        return res
