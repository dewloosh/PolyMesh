# -*- coding: utf-8 -*-
"""
These base classes meant to resolve circular references
while providing static hints.

This module must not have references from other parts of the library,
to make sure circular refrerences are all avoided.

"""
from abc import abstractmethod, abstractproperty

from numpy import ndarray

from ..core import DeepDict

from ..mesh.topo import TopologyArray

from .akwrap import AkWrapper
from .abc import ABC_MeshData


class PointDataBase(AkWrapper, ABC_MeshData):

    @abstractproperty
    def id(self) -> ndarray:
        """Ought to return global ids of the points."""
        ...

    @abstractproperty
    def frame(self) -> ndarray:
        """Ought to return a frame of reference."""
        ...

    @abstractproperty
    def x(self) -> ndarray:
        """Ought to return the coordinates of the associated pointcloud."""
        ...


class CellDataBase(AkWrapper, ABC_MeshData):

    @abstractproperty
    def id(self) -> ndarray:
        """Ought to return global ids of the cells."""
        ...

    @abstractmethod
    def coords(self, *args, **kwargs) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    @abstractmethod
    def topology(self, *args, **kwargs) -> TopologyArray:
        """Ought to return the topology associated with the object."""
        ...

    @abstractmethod
    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...

    @abstractmethod
    def measure(self, *args, **kwargs) -> ndarray:
        """Ought to return a single measure for a collection of cells."""
        ...
        
    def to_triangles(self, *args, **kwargs) -> ndarray:
        """Ought to return a triangular representation of the mesh."""
        raise NotImplementedError


class PolyDataBase(DeepDict):

    @abstractmethod
    def coords(self, *args, **kwargs) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    @abstractmethod
    def topology(self, *args, **kwargs) -> TopologyArray:
        """Ought to return the topology associated with the object."""
        ...
