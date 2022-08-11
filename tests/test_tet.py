# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dewloosh.geom import TriMesh, CartesianFrame
from dewloosh.geom.primitives import circular_disk
from dewloosh.geom.cells import T3


class TestTet(unittest.TestCase):

    def test_vol_TET4(self):
        def test_vol_TET4(Lx, Ly, Lz, nx, ny, nz):
            try:
                A = CartesianFrame(dim=3)
                mesh2d = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T3)
                mesh3d = mesh2d.extrude(h=Lz, N=nz)
                assert np.isclose(mesh3d.volume(), Lx*Ly*Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e
        assert test_vol_TET4(1.0, 1.0, 1.0, 2, 2, 2)
        
    def test_vol_cylinder_TET4(self):
        def test_vol_cylinder_TET4(min_radius, max_radius, height, 
                           n_angles, n_radii, n_z):
            try:
                A = CartesianFrame(dim=3)
                points, triangles = \
                    circular_disk(n_angles, n_radii, min_radius, max_radius)
                mesh2d = TriMesh(points=points, triangles=triangles, frame=A)
                mesh3d = mesh2d.extrude(h=height, N=n_z)
                a = np.pi * (max_radius**2 - min_radius**2) * height
                assert np.isclose(mesh3d.volume(), a, atol=0, rtol=a/1000)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e
        assert test_vol_cylinder_TET4(1., 10., 10., 120, 80, 5)


if __name__ == "__main__":
            
    unittest.main()
