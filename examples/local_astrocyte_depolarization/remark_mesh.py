#!/usr/bin/python3

import dolfinx
from mpi4py import MPI
import numpy as np
import scifem

comm = MPI.COMM_WORLD

# Region in which to apply the source term (cm)
x_L = 2100e-7; x_U = 2900e-7
y_L = 2100e-7; y_U = 2900e-7
z_L = 2100e-7; z_U = 2500e-7

def print_coordinates(coordinates, domain, domain_prefix):
    # list of point in domain and ROI
    points = []
    for i in coordinates:
        for j in i:
            x = j[0]
            y = j[1]
            z = j[2]
            if x_L < x < x_U and y_L < y < y_U and z_L < z < z_U:
               points.append([x, y, z])

    middle = int(round((len(points) - 1)/2, 0))

    point = points[middle]
    x = point[0]
    y = point[1]
    z = point[2]

    print(f"Coordinates of point in {domain}")
    print(f"x_{domain_prefix} = {x}")
    print(f"y_{domain_prefix} = {y}")
    print(f"z_{domain_prefix} = {z}")
    print("-----------------------------------")

    point = points[middle-5]
    x = point[0]
    y = point[1]
    z = point[2]

    print(f"Coordinates of point in {domain}")
    print(f"x_{domain_prefix} = {x}")
    print(f"y_{domain_prefix} = {y}")
    print(f"z_{domain_prefix} = {z}")
    print("-----------------------------------")

    point = points[middle+5]
    x = point[0]
    y = point[1]
    z = point[2]

    print(f"Coordinates of point in {domain}")
    print(f"x_{domain_prefix} = {x}")
    print(f"y_{domain_prefix} = {y}")
    print(f"z_{domain_prefix} = {z}")
    print("-----------------------------------")

if __name__ == "__main__":

    # Get mesh
    mesh_path = 'meshes/envelopsize+18/'
    mesh_file = mesh_path + "mesh.xdmf"
    fcts_file = mesh_path + "facets.xdmf"

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    # Read mesh and cell tags
    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode)
        ct_org = xdmf.read_meshtags(mesh, name='mesh')

    xdmf.close()

    # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
    mesh.topology.create_entities(mesh.topology.dim-1)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    # get topological dimension
    tdim = mesh.topology.dim

    # Get total number of cells
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts

    # Set all markers to the default '1' first
    cell_marker = np.full(num_cells_local, 1, dtype=np.int32)

    # Get values and indices of the original cell mesh tags
    tags = ct_org.values
    indices = ct_org.indices

    # Retag ECS and glial mesh-cells
    cell_marker[indices[tags == 1]] = 0     # ECS (old tag 1, new tag 0)
    cell_marker[indices[tags == 100]] = 2   # glial cell (old tag 100, new tag 2)

    # Create mesh tag function
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    )
    ct.name = "cell_marker"

    # Read facet tags
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fcts_file, 'r') as xdmf:
        ft_org = xdmf.read_meshtags(mesh, name="mesh")
    xdmf.close()

    # Get total number of facets
    facet_map = mesh.topology.index_map(tdim - 1)
    num_facets_local = facet_map.size_local + facet_map.num_ghosts

    # Mark all facets with interior facets marker (needed for the DG method)
    facet_marker = np.full(num_facets_local, 1, dtype=np.int32)

    # Get original cell and facet tags
    ct_org_tags = np.unique(ct_org.values)
    ft_org_tags = np.unique(ft_org.values)

    # Create new facet marker list (default value is 1)
    facet_marker = np.full(ft_org.values.shape, 1, dtype=np.int32)
    tags = ft_org.values

    # Get exterior boundary tag (facet tags not present in cell tags)
    exterior_ft = np.setdiff1d(ft_org_tags, ct_org_tags)
    exterior_tag = max(exterior_ft) if exterior_ft.size > 0 else 0
    # Check if facet is exterior facet
    is_exterior = np.isin(tags, exterior_ft) & (tags != 0)

    # Re-mark facets
    facet_marker[tags == 100] = 2              # Glial facets (old tag 100, new tag 2)
    facet_marker[tags == 0] = 0              # ECS facets (old tag 0, new tag 0)
    facet_marker[is_exterior] = exterior_tag # Exterior facets

    # Create new facet markers
    ft = dolfinx.mesh.meshtags(
        mesh, tdim - 1, np.arange(num_facets_local, dtype=np.int32), facet_marker
    )
    ft.name = "facet_marker"

    # Convert mesh from nm to cm
    mesh.geometry.x[:] *= 1e-7

    output_path = "meshes/remarked_mesh/"
    # Define the output filename
    xdmf_filename = f"{output_path}mesh.xdmf"

    # Create an XDMFFile object in write mode ('w')
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf_filename, "w") as xdmf:
        # Write the mesh and cell tags to the XDMF file
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)

    xdmf.close()

    # Get the indices of facets with tag value 1
    tagged_facet_indices = ft.find(1)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ft.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "membrane neuron", "M")

    tagged_facet_indices = ct.find(1)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ct.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "ICS neuron", "i")

    tagged_facet_indices = ct.find(0)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ct.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "ECS neuron", "e")

    # Get the indices of facets with tag value 2
    tagged_facet_indices = ft.find(2)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ft.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "membrane glial", "M")

    tagged_facet_indices = ct.find(2)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ct.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "ICS glial", "i")

    tagged_facet_indices = ct.find(0)
    vertex_indices = dolfinx.mesh.entities_to_geometry(mesh, ct.dim, tagged_facet_indices)
    coordinates = mesh.geometry.x[vertex_indices]
    print_coordinates(coordinates, "ECS glial", "e")
