# -*- coding: utf-8 -*-
import numpy as np
import unittest

from polymesh.trimesh import TriMesh
from polymesh.grid import Grid
from polymesh import PolyData
from polymesh.recipes import circular_disk
from polymesh.voxelize import voxelize_cylinder
from polymesh.cells import H8, TET4
from polymesh.utils.topology import detach_mesh_bulk
from polymesh.extrude import extrude_T3_TET4
from polymesh.space import StandardFrame


class TestMeshing(unittest.TestCase):
    def test_trimesh(self):
        trimesh = TriMesh(size=(800, 600), shape=(10, 10))
        disk = circular_disk(120, 60, 5, 25)

    def test_grid(self):
        Grid(size=(80, 60, 20), shape=(8, 6, 2), eshape="H8")
        Grid(size=(80, 60, 20), shape=(8, 6, 2), eshape="H27")

    def test_voxelize(self):
        d, h, a, b = 100.0, 0.8, 1.5, 0.5
        coords, topo = voxelize_cylinder(radius=[b / 2, a / 2], height=h, size=h / 20)
        PolyData(coords=coords, topo=topo, celltype=H8)

    def test_extrude(self):
        n_angles = 120
        n_radii = 60
        min_radius = 5
        max_radius = 25
        h = 20
        zres = 20
        mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
        points = mesh.coords()
        triangles = mesh.topology()
        points, triangles = detach_mesh_bulk(points, triangles)
        coords, topo = extrude_T3_TET4(points, triangles, h, zres)

        A = StandardFrame(dim=3)
        tetmesh = PolyData(coords=coords, topo=topo, celltype=TET4, frame=A)

        trimesh = TriMesh(size=(800, 600), shape=(10, 10), frame=A)
        trimesh.area()
        tetmesh = trimesh.extrude(h=300, N=5)
        tetmesh.volume()

    def test_tet(self):
        trimesh = TriMesh(size=(800, 600), shape=(10, 10))
        tetmesh = trimesh.extrude(h=300, N=5)
        assert tetmesh.volume() == 144000000.0


if __name__ == "__main__":
    unittest.main()
