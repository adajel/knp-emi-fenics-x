import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
from scipy.spatial import cKDTree as KDTree
import meshio

def map_kdtree(data_points, query_points, **kwargs):
    tree = KDTree(data_points)
    dist, idx = tree.query(query_points, **kwargs)
    return dist, idx

def compute_local_width(mesh, ecs_id, labelname="label", width_bins=None):
    ecs = mesh.extract_cells(np.isin(mesh.cell_data[labelname], ecs_id))
    ecs_surf = ecs.extract_surface()
    # compute distance from the mebrane at a set of points
    points = np.vstack([ecs.cell_centers().points, ecs.points])
    pointset = pv.PointSet(points).compute_implicit_distance(ecs_surf)
    cell_midpoints = ecs.cell_centers().points
    dist = abs(pointset["implicit_distance"])

    # set a discrete set of width bins (by default width is between 0 and the miximum distance to the surface)
    if width_bins is None:
        width_bins = np.linspace(0, dist.max(), 50, endpoint=False)

    # initiate the local width to zero
    ecs["local_width"] = np.zeros(ecs.number_of_cells)

    # now, iterate over the width bins
    for ri in np.array(width_bins) / 2:
        # identify all points that are center points of a ball of radius= width/2 that fits into the space
        mask = dist>=ri
        # next, find all cells with midpoint in one of those balls - they have that local width!
        if mask.sum() > 0:
            current_dist, idx = map_kdtree(points[mask], cell_midpoints, distance_upper_bound=ri)
            ecs["local_width"] = np.maximum(ecs["local_width"],  2*ri*(current_dist<ri))
    return ecs

#grid = pv.read("grid.vtk")
mesh = meshio.read("meshes/mesh_size_5000/mesh.xdmf")
meshio.write("mesh.vtk", mesh)
grid = pv.read("mesh.vtk")

ecs = compute_local_width(grid, ecs_id=1).cell_data_to_point_data(pass_cell_data=True)
ecs.save("ecs.vtk")

ecs = compute_local_width(grid, ecs_id=100).cell_data_to_point_data(pass_cell_data=True)
ecs.save("glial.vtk")
