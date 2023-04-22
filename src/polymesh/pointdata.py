from typing import Union

import numpy as np
from numpy import ndarray

from dewloosh.core import classproperty
from neumann.linalg import ReferenceFrame as FrameLike
from neumann.logical import isboolarray
from neumann.linalg.sparse import csr_matrix

from .space import CartesianFrame, PointCloud
from .base import PointDataBase, PolyDataBase as PolyData
from .utils import collect_nodal_data


def gen_frame(coords):
    return CartesianFrame(dim=coords.shape[1])


class PointData(PointDataBase):
    """
    A class to handle data related to the pointcloud of a polygonal mesh.

    Technicall this is a wrapper around an :class:`awkward.Record` instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is important to understand how it works.
    """

    _point_cls_ = PointCloud
    _frame_class_ = CartesianFrame
    _attr_map_ = {
        "x": "x",  # coordinates
        "activity": "activity",  # activity of the points
        "id": "id",  # global indices of the points
    }

    def __init__(
        self,
        *args,
        points=None,
        coords=None,
        wrap=None,
        fields=None,
        frame: FrameLike = None,
        newaxis: int = 2,
        activity=None,
        db=None,
        container: PolyData = None,
        **kwargs
    ):
        if db is not None:
            wrap = db
        elif wrap is not None:
            pass
        else:
            fields = {} if fields is None else fields
            assert isinstance(fields, dict)

            # coordinate frame
            if not isinstance(frame, FrameLike):
                if coords is not None:
                    frame = gen_frame(coords)
            self._frame = frame

            # set pointcloud
            point_cls = self.__class__._point_cls_
            X = None
            if len(args) > 0:
                if isinstance(args[0], np.ndarray):
                    X = args[0]
            else:
                X = points if coords is None else coords
            assert isinstance(
                X, np.ndarray
            ), "Coordinates must be specified as a numpy array!"
            nP, nD = X.shape
            if nD == 2:
                inds = [0, 1, 2]
                inds.pop(newaxis)
                if isinstance(frame, FrameLike):
                    if len(frame) == 3:
                        _c = np.zeros((nP, 3))
                        _c[:, inds] = X
                        X = _c
                        X = point_cls(X, frame=frame).show()
                    elif len(frame) == 2:
                        X = point_cls(X, frame=frame).show()
            elif nD == 3:
                X = point_cls(X, frame=frame).show()
            fields[self._dbkey_x_] = X

            if activity is None:
                activity = np.ones(nP, dtype=bool)
            else:
                assert (
                    isboolarray(activity) and len(activity.shape) == 1
                ), "'activity' must be a 1d boolean numpy array!"
            fields[self._dbkey_activity_] = activity

            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == nP:
                        fields[k] = v

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)
        self._container = container

    @classproperty
    def _dbkey_id_(cls) -> str:
        return cls._attr_map_["id"]

    @classproperty
    def _dbkey_x_(cls) -> str:
        return cls._attr_map_["x"]

    @classproperty
    def _dbkey_activity_(cls) -> str:
        return cls._attr_map_["activity"]

    @property
    def has_id(self) -> ndarray:
        return self._dbkey_id_ in self._wrapped.fields

    @property
    def has_x(self) -> ndarray:
        return self._dbkey_x_ in self._wrapped.fields

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

    @property
    def frame(self) -> FrameLike:
        """
        Returns the frame of the underlying pointcloud.
        """
        if isinstance(self._frame, FrameLike):
            return self._frame
        else:
            dim = self.x.shape[-1]
            return self._frame_class_(dim=dim)

    @property
    def activity(self) -> ndarray:
        return self._wrapped[self._dbkey_activity_].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_activity_] = value

    @property
    def x(self) -> ndarray:
        return self._wrapped[self._dbkey_x_].to_numpy()

    @x.setter
    def x(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_x_] = value

    @property
    def id(self) -> ndarray:
        return self._wrapped[self._dbkey_id_].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_id_] = value

    def pull(self, key: str, ndf: Union[ndarray, csr_matrix] = None) -> ndarray:
        """
        Pulls data from the cells in the model. The pulled data is either copied or
        distributed according to a measure.

        Parameters
        ----------
        key: str
            A field key to identify data in the databases of the attached
            CellData instances of the blocks.

        See Also
        --------
        :func:`~polymesh.utils.utils.collect_nodal_data`
        """
        source = self.container
        if ndf is None:
            ndf = source.nodal_distribution_factors()
        if isinstance(ndf, ndarray):
            ndf = csr_matrix(ndf)
        blocks = list(source.cellblocks(inclusive=True))
        b = blocks.pop(0)
        cids = b.cd.id
        topo = b.cd.nodes
        celldata = b.cd.db[key].to_numpy()
        if len(celldata.shape) == 1:
            nE, nNE = topo.shape
            celldata = np.repeat(celldata, nNE).reshape(nE, nNE)
        shp = [len(self)] + list(celldata.shape[2:])
        res = np.zeros(shp, dtype=float)
        collect_nodal_data(celldata, topo, cids, ndf, res)
        for b in blocks:
            cids = b.cd.id
            topo = b.cd.nodes
            celldata = b.cd.db[key].to_numpy()
            if len(celldata.shape) == 1:
                nE, nNE = topo.shape
                celldata = np.repeat(celldata, nNE).reshape(nE, nNE)
            collect_nodal_data(celldata, topo, cids, ndf, res)
        return res
