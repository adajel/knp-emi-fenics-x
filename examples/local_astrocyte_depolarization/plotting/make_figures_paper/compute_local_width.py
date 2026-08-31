import numpy as np
import pyvista as pv
import meshio
from scipy.spatial import cKDTree as KDTree

def map_kdtree(data_points, query_points, **kwargs):
    tree = KDTree(data_points)
    dist, _ = tree.query(query_points, **kwargs)
    return dist

def compute_local_width(mesh, domain_id, labelname="marker", width_bins=None):
    # Ensure tag array data is integer format for np.isin
    tags = mesh.cell_data[labelname].astype(np.int64)

    # Extract subdomain cells
    ecs = mesh.extract_cells(np.isin(tags, domain_id))
    ecs_surf = ecs.extract_surface(algorithm="dataset_surface")

    # Compute implicit distance to surface
    cell_midpoints = ecs.cell_centers().points
    points = np.vstack([cell_midpoints, ecs.points])

    pointset = pv.PointSet(points).compute_implicit_distance(ecs_surf)
    dist = np.abs(pointset["implicit_distance"])

    if width_bins is None:
        width_bins = np.linspace(0, dist.max(), 50, endpoint=False)

    local_widths = np.zeros(ecs.number_of_cells)

    for ri in np.array(width_bins) / 2:
        mask = dist >= ri
        if mask.sum() > 0:
            current_dist = map_kdtree(points[mask], cell_midpoints, distance_upper_bound=ri)
            local_widths = np.maximum(local_widths, 2 * ri * (current_dist < ri))

    ecs["local_width"] = local_widths
    return ecs

def main():
    datasets = [("D1", 3), ("D2", 3), ("D3", 3)]

    for dname, glial_id in datasets:
        filename = f"../../meshes/synapse_{dname}/meshes/mesh.xdmf"

        # Read directly using meshio to PyVista (bypasses VTK write/read errors)
        msh = meshio.read(filename)
        grid = pv.from_meshio(msh)

        # Using 'marker' as listed in your output's dict_keys(['marker', 'label'])
        labelname = "marker" if "marker" in grid.cell_data else list(grid.cell_data.keys())[0]

        # Compute and export local width meshes
        ecs = compute_local_width(grid, domain_id=1, labelname=labelname)
        ecs.cell_data_to_point_data(pass_cell_data=True).save(f"results/ecs_{dname}.vtk")

        glial = compute_local_width(grid, domain_id=glial_id, labelname=labelname)
        glial.cell_data_to_point_data(pass_cell_data=True).save(f"results/glial_{dname}.vtk")

if __name__ == "__main__":
    main()
