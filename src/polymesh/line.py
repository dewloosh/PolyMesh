# -*- coding: utf-8 -*-
from .utils.utils import lengths_of_lines
from .cell import PolyCell1d
import numpy as np
from numpy import ndarray

from .utils.utils import jacobian_matrix_bulk_1d, jacobian_det_bulk_1d


__all__ = ["Line", "QuadraticLine", "NonlinearLine"]


class Line(PolyCell1d):
    """
    Base class for all lines.

    """

    NNODE = 2
    vtkCellType = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lengths(self, *args, coords=None, topo=None, **kwargs) -> ndarray:
        """
        Returns the lengths as a NumPy array.
        """
        if coords is None:
            coords = self.root().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        return lengths_of_lines(coords, topo)

    def areas(self, *args, **kwargs) -> ndarray:
        """
        Returns the areas as a NumPy array.
        """
        areakey = self._dbkey_areas_
        if areakey in self.fields:
            return self[areakey].to_numpy()
        else:
            return np.ones((len(self)))

    def volumes(self, *args, **kwargs):
        """
        Returns the volumes as a NumPy array.
        """
        return self.lengths(*args, **kwargs) * self.areas(*args, **kwargs)

    def jacobian_matrix(self, *args, dshp: ndarray = None, **kwargs):
        """
        Calculates jacobian matrices.

        Parameters
        ----------
        dshp : numpy.ndarray
            Array of shape function derivatives.
        """
        assert dshp is not None
        ecoords = kwargs.get("ecoords", self.local_coordinates())
        return jacobian_matrix_bulk_1d(dshp, ecoords)

    def jacobian(self, *args, jac: ndarray = None, **kwargs):
        """
        Calculates jacobian determinants.

        Parameters
        ----------
        jac : numpy.ndarray
            Array of jacobian matrices derivatives.

        See Also
        --------
        :func:`jacobian_matrix`
        """
        return jacobian_det_bulk_1d(jac)


class QuadraticLine(Line):
    """
    Base class for quadratic 3-noded lines.

    See Also
    --------
    :class:`Line`
    """

    NNODE = 3
    vtkCellType = None


class NonlinearLine(Line):
    """
    Base class for general nonlinear lines.

    See Also
    --------
    :class:`Line`
    """

    NNODE: int = None
    vtkCellType = None
