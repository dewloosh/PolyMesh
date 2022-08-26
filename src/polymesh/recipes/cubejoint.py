# -*- coding: utf-8 -*-
import numpy as np

import tetgen

from sectionproperties.analysis.section import Section

from neumann.linalg import linspace, Vector

from polymesh import PolyData
from polymesh.grid import grid
from polymesh.tri.triang import triangulate
from polymesh.space import StandardFrame, PointCloud
from polymesh.utils import centralize, center_of_points
from polymesh.tri.triutils import get_points_inside_triangles, \
    approx_data_to_points
from polymesh.topo import remap_topo
from polymesh.topo.tr import T6_to_T3

from sigmaepsilon.solid import Structure, FemMesh
from sigmaepsilon.solid.fem.cells import TET4, H8


nodes_of_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]
corners_of_faces = {
    'front': [1, 2, 6, 5],
    'back': [0, 3, 7, 4],
    'left': [2, 3, 7, 6],
    'right': [0, 1, 5, 4],
    'bottom': [0, 1, 2, 3],
    'top': [4, 5, 6, 7],
}
edges_of_faces = {
    'front': [1, 5, 9, 10],
    'back': [3, 7, 8, 11],
    'right': [0, 9, 4, 8],
    'left': [2, 6, 10, 11],
    'bottom': [0, 1, 2, 3],
    'top': [4, 5, 6, 7],
}


def generate_frames(size, points_per_edge):
    points = []
    Lx, Ly, Lz = size
    # corners
    corner_coords = [
        [-Lx/2, -Ly/2, -Lz/2],
        [Lx/2, -Ly/2, -Lz/2],
        [Lx/2, Ly/2, -Lz/2],
        [-Lx/2, Ly/2, -Lz/2],
        [-Lx/2, -Ly/2, Lz/2],
        [Lx/2, -Ly/2, Lz/2],
        [Lx/2, Ly/2, Lz/2],
        [-Lx/2, Ly/2, Lz/2]
    ]
    corner_coords = np.array(corner_coords)
    points.append(corner_coords)

    # populate edges
    edge_coords = []
    N = points_per_edge + 2
    for nodes in nodes_of_edges:
        p0 = corner_coords[nodes[0]]
        p1 = corner_coords[nodes[1]]
        edge_coords.append(linspace(p0, p1, N)[1:-1])
    edge_coords = np.vstack(edge_coords)
    points.append(edge_coords)

    # center of face
    def cof(id):
        return center_of_points(corner_coords[corners_of_faces[id]])

    # face frames
    frames = {}
    GlobalFrame = StandardFrame(dim=3)
    frames['front'] = GlobalFrame.fork().move(cof('front'))
    rot90z = 'Body', [0, 0, np.pi/2], 'XYZ'
    frames['left'] = frames['front'].fork(
        *rot90z).move(cof('left') - cof('front'))
    frames['back'] = frames['left'].fork(
        *rot90z).move(cof('back') - cof('left'))
    frames['right'] = frames['back'].fork(
        *rot90z).move(cof('right') - cof('back'))
    rot_front_top = 'Body', [0, -np.pi/2, 0], 'XYZ'
    frames['top'] = frames['front'].fork(
        *rot_front_top).move(cof('top') - cof('front'))
    rot180y = 'Body', [0, np.pi, 0], 'XYZ'
    frames['bottom'] = frames['top'].fork(
        *rot180y).move(cof('bottom') - cof('top'))

    return points, frames


def joint_cube_voxelize(size, shape, *args, sections_dict=None,
                        material=None, **kwargs) -> Structure:
    points_per_edge = shape + 1
    mesh_size = min(size) / (points_per_edge-1)
    E = material['E']
    nu = material['nu']

    GlobalFrame = StandardFrame(dim=3)
    coords, topo = grid(size=size, shape=(shape, shape, shape),
                        eshape='H8', centralize=True)
    fixity = np.zeros_like(coords).astype(bool)
    loads = np.zeros_like(coords).astype(float)
    coords = PointCloud(coords, frame=GlobalFrame)
    _, frames = generate_frames(size, points_per_edge)

    for face in sections_dict:
        f_frame = frames[face]
        coords_ = coords.show(f_frame)
        f_inds = np.where(np.abs(coords_[:, 0]) < 1e-8)[0]
        f_coords = coords_[f_inds, 1:]
        f_section = sections_dict[face]['geom']
        f_section.create_mesh(mesh_sizes=[mesh_size])
        f_coords_s = centralize(np.array(f_section.mesh['vertices']))
        f_topo_s = np.array(f_section.mesh['triangles'])

        # boundary conditions
        if 'dynams' in sections_dict[face]:
            dyn = sections_dict[face]['dynams']
            _section = Section(f_section)
            _section.calculate_geometric_properties()
            _section.calculate_warping_properties()
            stress_post = _section.calculate_stress(
                N=dyn['N'], Vy=dyn['Vy'], Vx=dyn['Vx'],
                Mzz=dyn['T'], Mxx=dyn['Mx'], Myy=dyn['My']
            )
            stresses = stress_post.get_stress()[0]
            f_data_s = np.zeros((len(f_coords_s), 3))
            f_data_s[:, 0] = stresses['sig_zz']
            f_data_s[:, 1] = stresses['sig_zx']
            f_data_s[:, 2] = stresses['sig_zy']
            f_data_s = Vector(f_data_s, frame=f_frame).show(GlobalFrame)
            f_coords_s, f_topo_s = T6_to_T3(f_coords_s, f_topo_s)
            f_data = approx_data_to_points(
                f_coords_s, f_topo_s, f_data_s, f_coords)
            loads[f_inds, :] = f_data
        elif 'support' in sections_dict[face]:
            cond = get_points_inside_triangles(
                f_coords_s, f_topo_s[:, :3], f_coords)
            f_inds = np.where(cond)[0]
            fixity[f_inds, :] = True
        else:
            raise NotImplementedError

    # Hooke model
    Hooke = np.array([
        [1, nu, nu, 0, 0, 0],
        [nu, 1, nu, 0, 0, 0],
        [nu, nu, 1, 0, 0, 0],
        [0., 0, 0, (1-nu)/2, 0, 0],
        [0., 0, 0, 0, (1-nu)/2, 0],
        [0., 0, 0, 0, 0, (1-nu)/2]]) * (E / (1-nu**2))

    # return finite element model
    mesh = FemMesh(coords=coords.array, topo=topo, celltype=H8,
                   model=Hooke, fixity=fixity, loads=loads)
    structure = Structure(mesh=mesh)

    return structure


def joint_cube(size, shape, *args, sections_dict=None, material=None,
               voxelize=False, varvolume=None, **kwargs) -> Structure:
    if voxelize:
        return joint_cube_voxelize(size, shape, *args,
                                   sections_dict=sections_dict,
                                   material=material, **kwargs)
    Lx, Ly, Lz = size
    points_per_edge = shape + 1
    mesh_size = min(size) / (points_per_edge-1)
    E = material['E']
    nu = material['nu']

    # base points
    points, frames = generate_frames(size, points_per_edge)
    points = [np.vstack(points), ]
    loads = [np.zeros((points[0].shape[0], 3)).astype(float), ]
    fixity = [np.zeros((points[0].shape[0], 3)).astype(bool), ]
    corner_coords = points[0][:8]
    edge_coords = points[0][8:]
    nTotalPoints = len(points[0])

    # background grid
    GlobalFrame = StandardFrame(dim=3)
    N = points_per_edge + 2
    coords_grid, topo_grid = grid(size=(Lx*0.99, Ly*0.99), shape=(N, N),
                                  eshape='Q4', centralize=True)
    Grid = PolyData(coords=coords_grid, topo=topo_grid, frame=GlobalFrame)
    grid_centers = Grid.centers()[:, :2]
    fixity_grid = np.zeros((grid_centers.shape[0], 3)).astype(bool)
    sig_grid = np.zeros((grid_centers.shape[0], 3)).astype(float)

    # loop over each face, add new points and triangulations
    for face in frames:
        f_frame = frames[face]
        # collect points on corners and edges and their global indices
        # these indices are used later to remap the default topology
        # resulting from per-face triangulations
        f_coords_base = []
        f_inds_base = []
        _corner_inds = []
        for corner in corners_of_faces[face]:
            f_coords_base.append(corner_coords[corner])
            _corner_inds.append(corner)
        f_inds_base.append(np.array(_corner_inds, dtype=int))
        for edge in edges_of_faces[face]:
            inds = np.arange(points_per_edge) + edge * points_per_edge
            f_coords_base.append(edge_coords[inds])
            f_inds_base.append(inds + 8)
        f_coords_base = np.vstack(f_coords_base)
        sig_coords_base = np.zeros((f_coords_base.shape[0], 3)).astype(float)
        fixity_coords_base = np.zeros((f_coords_base.shape[0], 3)).astype(bool)

        # transform the coords (base coordinates) so far to face frame
        f_coords_base = PointCloud(
            f_coords_base, frame=GlobalFrame).show(f_frame)
        #f_coords_base = yz_to_xy(f_coords_base)[:, :2]

        # global indices and number of corner and edge points
        f_inds_base = np.concatenate(f_inds_base)
        nBasePoints = len(f_inds_base)

        # build face
        if face in sections_dict:
            # 1) create the mesh of the section
            # 2) rule out points of the base grid that the section covers
            # 3) add corner and edge nodes and do a triangulation
            f_section = sections_dict[face]['geom']
            f_section.create_mesh(mesh_sizes=[mesh_size])
            f_coords = centralize(np.array(f_section.mesh['vertices']))

            # boundary conditions
            sig_coords = np.zeros((f_coords.shape[0], 3)).astype(float)
            fixity_coords = np.zeros((f_coords.shape[0], 3)).astype(bool)
            if 'dynams' in sections_dict[face]:
                dyn = sections_dict[face]['dynams']
                _section = Section(f_section)
                _section.calculate_geometric_properties()
                _section.calculate_warping_properties()
                stress_post = _section.calculate_stress(
                    N=dyn['N'], Vy=dyn['Vy'], Vx=dyn['Vx'],
                    Mzz=dyn['T'], Mxx=dyn['Mx'], Myy=dyn['My']
                )
                stresses = stress_post.get_stress()
                sig_coords[:, 0] = stresses[0]['sig_zz']
                sig_coords[:, 1] = stresses[0]['sig_zx']
                sig_coords[:, 2] = stresses[0]['sig_zy']
                sig_coords = Vector(
                    sig_coords, frame=f_frame).show(GlobalFrame)
            elif 'support' in sections_dict[face]:
                fixity_coords[:, :] = True
            else:
                raise NotImplementedError

            n_section_nodes = f_coords.shape[0]
            f_topo = np.array(f_section.mesh['triangles'].tolist())[:, :3]
            f_inds = get_points_inside_triangles(
                f_coords, f_topo, grid_centers)
            f_coords = np.vstack(
                [f_coords_base[:, 1:], f_coords, grid_centers[~f_inds]])
            f_coords, f_topo, _ = triangulate(points=f_coords)
            f_sig = np.vstack([sig_coords_base, sig_coords, sig_grid[~f_inds]])
            f_fixity = np.vstack(
                [fixity_coords_base, fixity_coords, fixity_grid[~f_inds]])
        else:
            f_coords = np.vstack([f_coords_base[:, 1:], grid_centers])
            f_coords, f_topo, _ = triangulate(points=f_coords)
            f_sig = np.zeros((f_coords.shape[0], 3)).astype(float)
            f_fixity = np.zeros((f_coords.shape[0], 3)).astype(bool)

        # faces share some points, hence they must be consistent
        # in node numbering --> remap topology to match indices
        # of corner and edge nodes
        f_inds = np.zeros(len(f_coords), dtype=int)
        nNewPoints = len(f_coords) - nBasePoints
        f_inds[:nBasePoints] = f_inds_base
        f_inds[nBasePoints:] = np.arange(nNewPoints) + nTotalPoints
        nTotalPoints += nNewPoints
        f_topo = remap_topo(f_topo, f_inds)

        # transform to global and append new data to total collection
        f_coords_new = np.zeros((nNewPoints, 3))
        f_coords_new[:, 1:] = f_coords[nBasePoints:]
        f_coords = PointCloud(f_coords_new, frame=f_frame).show(GlobalFrame)
        points.append(f_coords)
        fixity.append(f_fixity[nBasePoints:])
        loads.append(f_sig[nBasePoints:])

        # add topology to face to total collection
        if face not in sections_dict:
            sections_dict[face] = {}
        sections_dict[face]['topo'] = f_topo

    # build the final cube shell
    cubepoints = np.vstack(points)
    nCubePoints = len(cubepoints)
    cube = PolyData(coords=cubepoints, frame=GlobalFrame)
    for face in frames:
        cube[face] = PolyData(topo=sections_dict[face]['topo'])

    # tetrahedralize
    switches="pa{}".format(varvolume)
    tet = tetgen.TetGen(cube.coords(), cube.topology())
    tet.tetrahedralize(order=1, mindihedral=10, minratio=1.1, 
                       quality=True, switches=switches)
    twtgrid = tet.grid
    coords = np.array(twtgrid.points).astype(float)
    topo = twtgrid.cells_dict[10].astype(int)

    # collect loads and supports
    loads_ = np.zeros((coords.shape[0], 3)).astype(float)
    loads_[:nCubePoints] = np.vstack(loads)
    fixity_ = np.zeros((coords.shape[0], 3)).astype(bool)
    fixity_[:nCubePoints] = np.vstack(fixity)

    # Hooke model
    Hooke = np.array([
        [1, nu, nu, 0, 0, 0],
        [nu, 1, nu, 0, 0, 0],
        [nu, nu, 1, 0, 0, 0],
        [0., 0, 0, (1-nu)/2, 0, 0],
        [0., 0, 0, 0, (1-nu)/2, 0],
        [0., 0, 0, 0, 0, (1-nu)/2]]) * (E / (1-nu**2))

    # return finite element model
    mesh = FemMesh(coords=coords, topo=topo, celltype=TET4,
                   model=Hooke, fixity=fixity_, loads=loads_)
    structure = Structure(mesh=mesh)

    return structure


if __name__ == '__main__':
    from sectionproperties.pre.library.steel_sections import circular_hollow_section as CHS
    from sectionproperties.pre.library.steel_sections import rectangular_hollow_section as RHS
    from sectionproperties.pre.library.steel_sections import i_section as ISection
    from sectionproperties.analysis.section import Section
    from sectionproperties.pre.pre import Material
    import numpy as np
    import pyvista as pv

    E = 200e3
    nu = 0.3
    steel = Material(
        name='Steel', elastic_modulus=E, poissons_ratio=nu, density=7.85e-6,
        yield_strength=250, color='grey'
    )

    sections_dict = {
        'left': {
            'geom': CHS(d=100, t=10, n=64),
            'support': True},
        'right': {
            'geom': RHS(d=100, b=100, t=10, r_out=0, n_r=0),
            'support': True},
        'front': {
            'geom': ISection(d=170, b=110, t_f=7.8, t_w=5.8, r=8.9, n_r=16, material=steel),
            'dynams': {'N': 1e3, 'Vx': 0., 'Vy': 3e3,
                       'T': 0.,  'Mx': 5e6, 'My': 0.}},
    }

    sections_dict = {
        'left': {
            'geom': CHS(d=100, t=10, n=64),
            'dynams': {'N': 0, 'Vx': 0., 'Vy': 0,
                       'T': 1e9,  'Mx': 0, 'My': 0.}
            },
        'right': {
            'geom': RHS(d=100, b=100, t=10, r_out=0, n_r=0),
            'support': True},
    }

    Lx, Ly, Lz = 150, 150, 150
    voxelize = False

    cube = joint_cube((Lx, Ly, Lz), 40, sections_dict=sections_dict,
                      material={'E': E, 'nu': nu}, voxelize=voxelize,
                      varvolume=60)

    cube.plot()
    coords = cube.coords()
    topo = cube.topology()
    cube.linsolve()

    data = cube.stresses_at_centers('HMH')
    centers = cube.centers()
    dofsol = cube.dofsol(flatten=False)
    
    mask = np.where(centers[:, 0] > 0)[0]

    p = pv.Plotter(notebook=False)
    celltype = H8 if voxelize else TET4
    cube2 = PolyData(coords=coords + dofsol, topo=topo[mask], celltype=celltype)
    p.add_mesh(cube2.to_pv(), scalars=data[mask], show_edges=True)
    p.add_mesh(cube.mesh.to_pv(), style='wireframe', color='black')
    p.show()
