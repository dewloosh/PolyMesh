# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from ..math.array import atleast2d, atleast3d, repeat

from .utils import avg_cell_data, distribute_nodal_data, \
    homogenize_nodal_values
from .base import PointDataBase, CellDataBase, PolyDataBase as PolyData


class CellData(CellDataBase):
    """
    A class to handle data related to the cells of a polygonal mesh.

    Technically this is a wrapper around an `awkward.Record` instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is important to understand how it works.

    """

    _attr_map_ = {
        'nodes': 'nodes',  # node indices
        'frames': 'frames',  # coordinate frames
        'ndf': 'ndf',  # nodal distribution factors
        'id': 'id',  # global indices of the cells
        'areas': 'areas',  # areas of 1d cells
        't': 't',  # thicknesses for 2d cells
        'activity': 'activity',  # activity of the cells
        'ndf': 'ndf',  # nodal distribution factors
    }

    def __init__(self, *args, pointdata=None, celldata=None,
                 wrap=None, topo=None, fields=None, frames=None,
                 db=None, activity=None, container: PolyData = None, **kwargs):
        amap = self.__class__._attr_map_
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)

        celldata = db if db is not None else celldata
        if celldata is not None:
            wrap = celldata
        else:
            if len(args) > 0:
                if isinstance(args[0], np.ndarray):
                    nodes = args[0]
            else:
                nodes = topo
            assert isinstance(nodes, np.ndarray)
            fields[amap['nodes']] = nodes

            if isinstance(activity, np.ndarray):
                fields[amap['activity']] = activity

            N = nodes.shape[0]
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == N:
                        fields[k] = v

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)
        
        self.pointdata = pointdata
        self._container = container
        
        if self.db is not None:
            if isinstance(frames, np.ndarray):
                # this handles possible repetition of a single frame
                self.frames = frames

    @property
    def pd(self) -> PointDataBase:
        return self.pointdata

    @pd.setter
    def pd(self, value: PointDataBase):
        self.pointdata = value

    @property
    def container(self) -> PolyData:
        return self._container

    @container.setter
    def container(self, value: PolyData):
        assert isinstance(value, PolyData)
        self._container = value

    def root(self) -> PolyData:
        c = self.container
        return None if c is None else c.root()

    def source(self) -> PolyData:
        c = self.container
        return None if c is None else c.source()

    def __getattr__(self, attr):
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
            raise AttributeError("'{}' object has no attribute \
                called {}".format(self.__class__.__name__, attr))
        except Exception:
            raise AttributeError("'{}' object has no attribute \
                called {}".format(self.__class__.__name__, attr))

    def set_nodal_distribution_factors(self, factors, key=None):
        if key is None:
            key = self.__class__._attr_map_['ndf']
        if len(factors) != len(self._wrapped):
            self._wrapped[key] = factors[self._wrapped.id]
        else:
            self._wrapped[key] = factors

    def pull(self, key: str = None, *args, ndfkey=None, store=False,
             storekey=None, avg=False, data=None, **kwargs):
        if ndfkey is None:
            ndfkey = self.__class__._attr_map_['ndf']
        storekey = key if storekey is None else storekey
        if key is not None:
            nodal_data = self.pointdata[key].to_numpy()
        else:
            assert isinstance(data, np.ndarray)
            nodal_data = data
        topo = self.nodes
        ndf = self._wrapped[ndfkey].to_numpy()
        if len(nodal_data.shape) == 1:
            nodal_data = atleast2d(nodal_data, back=True)
        d = distribute_nodal_data(nodal_data, topo, ndf)
        # nE, nNE, nDATA
        if isinstance(avg, np.ndarray):
            assert len(avg.shape) == 1
            assert avg.shape[0] == d.shape[0]
            d = homogenize_nodal_values(d, avg)
            # nE, nDATA
        d = np.squeeze(d)
        if store:
            self._wrapped[key] = d
        return d

    def spull(self, *args, storekey=None, **kwargs):
        return self.pull(*args, store=True, storekey=storekey, **kwargs)

    def push(self, *args, **kwargs):
        raise NotImplementedError()

    def spush(self, *args, storekey=None, **kwargs):
        return self.push(*args, store=True, storekey=storekey, **kwargs)

    @property
    def fields(self):
        return self._wrapped.fields

    @property
    def nodes(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['nodes']].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['nodes']] = value

    @property
    def frames(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['frames']].to_numpy()

    @frames.setter
    def frames(self, value: ndarray):
        assert isinstance(value, ndarray)
        value = atleast3d(value)
        if len(value) == 1:
            value = repeat(value[0], len(self.celldata._wrapped))
        else:
            assert len(value) == len(self._wrapped)
        self._wrapped[self.__class__._attr_map_['frames']] = value

    @property
    def id(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['id']].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['id']] = value

    @property
    def activity(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['activity']].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['activity']] = value
