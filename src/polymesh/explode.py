# -*- coding: utf-8 -*-
from numpy import ndarray

from .utils import explode_mesh_bulk, explode_mesh_data_bulk


def explode_mesh(coords: ndarray, topo: ndarray, *args, data=None, **kwargs):
    if data is None:
        return explode_mesh_bulk(coords, topo)
    elif isinstance(data, ndarray):
        return explode_mesh_data_bulk(coords, topo, data)
    else:
        raise NotImplementedError