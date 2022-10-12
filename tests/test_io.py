# -*- coding: utf-8 -*-
import numpy as np
import unittest

from polymesh.examples import stand_vtk
from neumann.array import minmax


class TestIO(unittest.TestCase):

    def test_io(self):
        mesh = stand_vtk(read=True)
        F = -10.0   
        
        mesh.centralize()
        mesh.move(np.array([0., 0., -mesh.coords()[:, 2].min()]))
        coords = mesh.coords()
        zmin, zmax = minmax(coords[:, 2])
        h = zmax - zmin
        zmin, zmax, h
        
        i_f = np.where(coords[:, 2]>=0.998*h)[0]
        i_u = np.where(coords[:, 2]<=0.002*h)[0]

        f = F * np.array([0., 0., 1, 0., 0., 0.]) / len(i_f)
        loads = np.zeros((coords.shape[0], 6), dtype=float)
        loads[i_f] = f

        fixity = np.zeros((coords.shape[0], 6), dtype=bool)
        fixity[i_u] = True
        
        mesh.pd['loads'] = loads
        mesh.pd['fixity'] = fixity
        mesh.pd.to_parquet('stand_pointdata.parquet')
        list(mesh.cellblocks())[0].cd.to_parquet('stand_celldata.parquet')
    


if __name__ == "__main__":
            
    unittest.main()
