# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from ..math.linalg import ReferenceFrame as FrameLike
from ..math.array import isboolarray

from .space import CartesianFrame, PointCloud
from .base import PointDataBase


def gen_frame(coords): return CartesianFrame(dim=coords.shape[1])


class PointData(PointDataBase):
    """
    A class to handle data related to the pointcloud of a polygonal mesh.

    Technicall this is a wrapper around an `awkward.Record` instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is important to understand how it works.

    """

    _point_cls_ = PointCloud
    _frame_class_ = CartesianFrame
    _attr_map_ = {
        'x': 'x',  # coordinates
        'activity': 'activity',  # activity of the points
        'id': 'id',  # global indices of the points
    }

    def __init__(self, *args, points=None, coords=None, wrap=None, fields=None,
                 frame: FrameLike = None, newaxis: int = 2, stateful=False,
                 activity=None, db=None, **kwargs):
        if db is not None:
            wrap = db
        elif wrap is not None:
            pass
        else:
            amap = self.__class__._attr_map_
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
                X, np.ndarray), 'Coordinates must be specified as a numpy array!'
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
            fields[amap['x']] = X

            if activity is None:
                activity = np.ones(nP, dtype=bool)
            else:
                assert isboolarray(activity) and len(activity.shape) == 1, \
                    "'activity' must be a 1d boolean numpy array!"
            if activity is None and stateful:
                fields[amap['active']] = np.ones(nP, dtype=bool)
            fields[amap['activity']] = activity

            if stateful:
                fields[amap['activity']] = np.ones(nP, dtype=bool)

            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == nP:
                        fields[k] = v

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)
    
    @property
    def frame(self) -> FrameLike:
        """Returns the frame of the underlying pointcloud."""
        if isinstance(self._frame, FrameLike):
            return self._frame
        else:
            dim = self.x.shape[-1]
            return self._frame_class_(dim=dim)

    @property
    def activity(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['activity']].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['activity']] = value

    @property
    def x(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['x']].to_numpy()

    @x.setter
    def x(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['x']] = value

    @property
    def id(self) -> ndarray:
        return self._wrapped[self.__class__._attr_map_['id']].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self.__class__._attr_map_['id']] = value
        
