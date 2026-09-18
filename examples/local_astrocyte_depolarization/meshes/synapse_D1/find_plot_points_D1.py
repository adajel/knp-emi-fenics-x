import os
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile

ghost_mode = dolfinx.mesh.GhostMode.shared_facet
comm = MPI.COMM_WORLD

mesh_file = "meshes/remarked_mesh_D1.xdmf"

with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
    # Read mesh and cell tags
    mesh = xdmf.read_mesh(ghost_mode=ghost_mode, name="Grid")
    ct = xdmf.read_meshtags(mesh, name='cell_marker')

    # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
    mesh.topology.create_entities(mesh.topology.dim-1)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    # Read facets
    ft = xdmf.read_meshtags(mesh, name='facet_marker')

xdmf.close()

# Define Region of Interest (ROI) boundaries in transformed spatial scale (cm)
roi = {
    "x_L": 2100.0e-7, "x_U": 2900.0e-7,
    "y_L": 2100.0e-7, "y_U": 2900.0e-7,
    "z_L": 2300.0e-7, "z_U": 2700.0e-7,
}

tdim = mesh.topology.dim      # 3D (Cells)
fdim = tdim - 1               # 2D (Facets/Surfaces)

f_to_v = mesh.topology.connectivity(fdim, 0)
c_to_v = mesh.topology.connectivity(tdim, 0)
coords = mesh.geometry.x

# Extract points by tag and filter by ROI
def get_roi_points(entity_indices, connectivity):
    points = {}
    for entity in entity_indices:
        nodes = connectivity.links(entity)
        c = coords[nodes]
        if np.all((c >= [roi["x_L"], roi["y_L"], roi["z_L"]]) & 
                  (c <= [roi["x_U"], roi["y_U"], roi["z_U"]])):
            for nid, coord in zip(nodes, c):
                points[nid] = coord * 1.0e7  # convert scale
    return points

membrane_points = get_roi_points(ft.indices[ft.values == 2], f_to_v)
intracellular_points = get_roi_points(ct.indices[ct.values == 2], c_to_v)
extracellular_points = get_roi_points(ct.indices[ct.values == 0], c_to_v)

# Print output
def print_points(name, points_dict):
    print(f"\n {name}")
    print(f" X | Y | Z")
    print("-" * 56)
    for nid, (x, y, z) in points_dict.items():
        x = round(x)
        y = round(y)
        z = round(z)
        # where ish we would like to get ECS point from
        #if 2400 < x < 2600 and 2400 < y < 2600: 
        # where ish we would like to get ICS and membrane point from
        if x < 2300 and 2700 < y < 2800 and 2400 < z < 2600:
            print(f"{x} | {y} | {z}")

print_points("Membrane Points (tag 2)", membrane_points)
print_points("Intracellular Points (tag 2)", intracellular_points)
print_points("Extracellular Points (tag 0)", extracellular_points)
