# -*- coding: utf-8 -*-
import numpy as np

from ..line import QuadraticLine


__all__ = ['L3']


class L3(QuadraticLine):

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., 0., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0.])

    """def shape_function_values(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return shp3_bulk(coords)
        else:
            return shp3(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp3_bulk(coords)
        else:
            return dshp3(coords)"""
