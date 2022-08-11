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
            return shp2_bulk(coords)
        else:
            return shp2(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp2_bulk(coords)
        else:
            return dshp2(coords)"""
