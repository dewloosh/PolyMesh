# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from dewloosh.core import classproperty
from neumann.array import atleast2d, atleast3d, repeat

from .base import (PointDataBase, CellDataBase, 
                   PolyDataBase as PolyData)
from .akwrap import AwkwardLike
from .utils import (avg_cell_data, distribute_nodal_data_bulk,
                    homogenize_nodal_values)


class CellData(CellDataBase):
    """
    A class to handle data related to the cells of a polygonal mesh.

    Technically this is a wrapper around an Awkward data object instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is useful to understand how it works.

    See Also
    --------
    :class:`awkward.Array`
    :class:`awkward.Record`
    """

    _attr_map_ = {
        'nodes': 'nodes',  # node indices
        'frames': 'frames',  # coordinate frames
        'ndf': 'ndf',  # nodal distribution factors
        'id': 'id',  # global indices of the cells
        'areas': 'areas',  # areas of 1d cells
        't': 't',  # thicknesses for 2d cells
        'activity': 'activity',  # activity of the cells
    }
    
    def __init__(self, *args, pointdata:PointDataBase=None, 
                 wrap:AwkwardLike=None, topo:ndarray=None, 
                 fields:dict=None, activity:ndarray=None,
                 frames:ndarray=None, db:AwkwardLike=None,  
                 container: PolyData = None, **kwargs):
        amap = self.__class__._attr_map_
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)

        if db is not None:
            wrap = db
        else:
            nodes = None
            if len(args) > 0:
                if isinstance(args[0], ndarray):
                    nodes = args[0]
            else:
                nodes = topo

            if isinstance(activity, ndarray):
                fields[amap['activity']] = activity

            if isinstance(nodes, ndarray):
                fields[amap['nodes']] = nodes
                N = nodes.shape[0]
                for k, v in kwargs.items():
                    if isinstance(v, ndarray):
                        if v.shape[0] == N:
                            fields[k] = v

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)

        self.pointdata = pointdata
        self._container = container

        if self.db is not None:
            if isinstance(frames, ndarray):
                # this handles possible repetition of a single frame
                self.frames = frames
    
    @classproperty
    def _dbkey_nodes_(cls) -> str:
        return cls._attr_map_['nodes']
    
    @classproperty
    def _dbkey_frames_(cls) -> str:
        return cls._attr_map_['frames']
    
    @classproperty
    def _dbkey_areas_(cls) -> str:
        return cls._attr_map_['areas']
    
    @classproperty
    def _dbkey_thickness_(cls) -> str:
        return cls._attr_map_['t']
    
    @classproperty
    def _dbkey_activity_(cls) -> str:
        return cls._attr_map_['activity']
    
    @classproperty
    def _dbkey_ndf_(cls) -> str:
        return cls._attr_map_['ndf']
    
    @classproperty
    def _dbkey_id_(cls) -> str:
        return cls._attr_map_['id']
    
    @property
    def has_id(self) -> ndarray:
        return self._dbkey_id_ in self._wrapped.fields
    
    @property
    def pd(self) -> PointDataBase:
        """
        Returns the attached point database. This is what
        the topology of the cells are referring to.
        """
        return self.pointdata

    @pd.setter
    def pd(self, value: PointDataBase):
        """
        Sets tje attached pointdata.
        """
        self.pointdata = value

    @property
    def container(self) -> PolyData:
        """
        Returns the container object of the block.
        """
        return self._container

    @container.setter
    def container(self, value: PolyData):
        """
        Sets the container of the block.
        """
        assert isinstance(value, PolyData)
        self._container = value

    def root(self) -> PolyData:
        """
        Returns the top level container of the model the block is
        the part of.
        """
        c = self.container
        return None if c is None else c.root()

    def source(self) -> PolyData:
        c = self.container
        return None if c is None else c.source()

    def __getattr__(self, attr):
        """
        Modified for being able to fetch data from pointcloud.
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        try:
            return getattr(self._wrapped, attr)
        except AttributeError:
            try:
                if self.pointdata is not None:
                    if attr in self.pointdata.fields:
                        data = self.pointdata[attr].to_numpy()
                        topo = self.nodes
                        return avg_cell_data(data, topo)
            except:
                pass
            name = self.__class__.__name__
            raise AttributeError(f"'{name}' object has no attribute called {attr}")
        except Exception:
            name = self.__class__.__name__
            raise AttributeError(f"'{name}' object has no attribute called {attr}")

    def set_nodal_distribution_factors(self, factors: ndarray, key: str = None):
        """
        Sets nodal distribution factors.

        Parameters
        ----------
        factors : numpy.ndarray
            A 3d float array. The length of the array must equal the number
            pf cells in the block.

        key : str, Optional
            A key used to store the values in the database. This makes you able
            to use more nodal distribution strategies in one model.
            If not specified, a default key is used.

        """
        if key is None:
            key = self.__class__._attr_map_['ndf']
        if len(factors) != len(self._wrapped):
            self._wrapped[key] = factors[self._wrapped.id]
        else:
            self._wrapped[key] = factors

    def pull(self, key: str = None, *args, ndfkey: str = None, store: bool = False,
             storekey: str = None, data: ndarray = None, distribute:bool=False, **kwargs):
        """
        Pulls data from the attached pointcloud. The pulled data is either copied or
        distributed according to a measure.

        Parameters
        ----------
        key : str, Optional
            A field key to identify data in the database of the attached pointcloud.
            If not specified, use the 'data' parameter to specify the data to pull.
            Default is None.
        ndfkey : str, Optional
            A field key to identify the distribution factors to use. If not specified,
            a default key is used. Default is None.
        store : bool, Optional
            Stores the pulled values in the database if True. If True, the pulled data
            is either stored with the same key used in the pointcloud, or a key specified
            with the parameter 'storekey'. Default is False.
        storekey : str, Optional
            A key used to store the values. If provided, the 'store' parameter is ignored.
            Default is False.
        data : numpy.ndarray, Optional
            Used to specify the data to pull, if the parameter 'key' is None.
            Default is None.
        distribute : bool, Optional
            If False, data is simply copied, otherwise it gets distributed according to
            the distribution factors of a measure. In the former case, parameters related
            to the distribution factors can be omitted. Default is False.
            
        See Also
        --------
        :func:`distribute_nodal_data_bulk`

        """
        if ndfkey is None and distribute:
            ndfkey = self.__class__._attr_map_['ndf']
        storekey = key if storekey is None else storekey
        if key is not None:
            nodal_data = self.pointdata[key].to_numpy()
        else:
            assert isinstance(data, ndarray), "No data to pull from!"
            nodal_data = data
        topo = self.nodes
        if distribute:
            ndf = self._wrapped[ndfkey].to_numpy()
        else:
            ndf = np.ones_like(topo).astype(float)
        if len(nodal_data.shape) == 1:
            nodal_data = atleast2d(nodal_data, back=True)
        d = distribute_nodal_data_bulk(nodal_data, topo, ndf)
        # nE, nNE, nDATA
        d = np.squeeze(d)
        if store:
            self._wrapped[key] = d
        return d

    def push(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def fields(self):
        """
        Returns the fields in the database object.

        """
        return self._wrapped.fields

    @property
    def nodes(self) -> ndarray:
        """
        Returns the topology of the cells.

        """
        return self._wrapped[self.__class__._attr_map_['nodes']].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray):
        """
        Sets the topology of the cells.

        Parameters
        ----------
        value : numpy.ndarray
            A 2d integer array.

        """
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['nodes']] = value

    @property
    def frames(self) -> ndarray:
        """
        Returns local coordinate frames of the cells.
        
        """
        return self._wrapped[self.__class__._attr_map_['frames']].to_numpy()

    @frames.setter
    def frames(self, value: ndarray):
        """
        Sets local coordinate frames of the cells.

        Parameters
        ----------
        value : numpy.ndarray
            A 3d float array.

        """
        assert isinstance(value, ndarray)
        value = atleast3d(value)
        if len(value) == 1:
            value = repeat(value[0], len(self._wrapped))
        else:
            assert len(value) == len(self._wrapped)
        self._wrapped[self.__class__._attr_map_['frames']] = value

    @property
    def id(self) -> ndarray:
        """
        Returns global indices of the cells.
        
        """
        return self._wrapped[self.__class__._attr_map_['id']].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        """
        Sets global indices of the cells.

        Parameters
        ----------
        value : numpy.ndarray
            An 1d integer array.

        """
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['id']] = value

    @property
    def activity(self) -> ndarray:
        """
        Returns a 1d boolean array of cell activity.
        
        """
        return self._wrapped[self.__class__._attr_map_['activity']].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        """
        Sets cell activity with a 1d boolean array.

        Parameters
        ----------
        value : numpy.ndarray
            An 1d bool array.

        """
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['activity']] = value
