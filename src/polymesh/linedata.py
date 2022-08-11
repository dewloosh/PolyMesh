# -*- coding: utf-8 -*-
import numpy as np

from .polydata import PolyData
from .cells import L2, L3
from .plotting.plotly import plot_lines_3d


__all__ = ['LineData']

 
class LineData(PolyData):
    
    _cell_classes_ = {
        2: L2,
        3: L3,
    }

    def __init__(self, *args, areas=None, **kwargs):          
        super().__init__(*args, **kwargs)
        if self.celldata is not None:
            nE = len(self.celldata)
            if areas is None:
                areas = np.ones(nE)
            else:
                assert len(areas.shape) == 1, \
                    "'areas' must be a 1d float or integer numpy array!"
            dbkey = self.celldata.__class__._attr_map_['areas']
            self.celldata.db[dbkey] = areas
            
    def _init_config_(self):
        super()._init_config_()
        key = self.__class__._pv_config_key_
        self.config[key]['color'] = 'k'
        self.config[key]['line_width'] = 10
        self.config[key]['render_lines_as_tubes'] = True 
        
    def plot(self, *args, scalars=None, backend='plotly', scalar_labels=None, **kwargs):
        if backend == 'vtk':
            return self.pvplot(*args, scalars=scalars, scalar_labels=scalar_labels, 
                               **kwargs)
        elif backend == 'plotly':
            return plot_lines_3d
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)
        
               