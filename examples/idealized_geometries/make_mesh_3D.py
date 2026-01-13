#!/usr/bin/env python3

"""
This script generates a 3D mesh representing 4 axons.
"""

import dolfinx
from mpi4py import MPI
import numpy as np
import scifem

l = 2

a_1 = [5e-6, 0.2e-6, 0.2e-6]
b_1 = [l * 16e-6 - 5e-6, 0.4e-6, 0.4e-6]

a_2 = [5e-6, 0.5e-6, 0.5e-6]
b_2 = [l * 16e-6 - 5e-6, 0.7e-6, 0.7e-6]

a_3 = [5e-6, 0.5e-6, 0.2e-6]
b_3 = [l * 16e-6 - 5e-6, 0.7e-6, 0.4e-6]

a_4 = [5e-6, 0.2e-6, 0.5e-6]
b_4 = [l * 16e-6 - 5e-6, 0.4e-6, 0.7e-6]

# lower left point and upper right point
x_L_1 = a_1[0]; y_L_1 = a_1[1]; z_L_1 = a_1[2]
x_U_1 = b_1[0]; y_U_1 = b_1[1]; z_U_1 = b_1[2]

# lower left point and upper right point
x_L_2 = a_2[0]; y_L_2 = a_2[1]; z_L_2 = a_2[2]
x_U_2 = b_2[0]; y_U_2 = b_2[1]; z_U_2 = b_2[2]

# lower left point and upper right point
x_L_3 = a_3[0]; y_L_3 = a_3[1]; z_L_3 = a_3[2]
x_U_3 = b_3[0]; y_U_3 = b_3[1]; z_U_3 = b_3[2]

# lower left point and upper right point
x_L_4 = a_4[0]; y_L_4 = a_4[1]; z_L_4 = a_4[2]
x_U_4 = b_4[0]; y_U_4 = b_4[1]; z_U_4 = b_4[2]

def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol

def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol

def mesh_interior_marker_1(x, tol=1e-12):
    return (
          lower_bound(x, 0, x_L_1, tol=tol)
        & lower_bound(x, 1, y_L_1, tol=tol)
        & lower_bound(x, 2, z_L_1, tol=tol)
        & upper_bound(x, 0, x_U_1, tol=tol)
        & upper_bound(x, 1, y_U_1, tol=tol)
        & upper_bound(x, 2, z_U_1, tol=tol)
    )

def mesh_interior_marker_2(x, tol=1e-12):
    return (
          lower_bound(x, 0, x_L_2, tol=tol)
        & lower_bound(x, 1, y_L_2, tol=tol)
        & lower_bound(x, 2, z_L_2, tol=tol)
        & upper_bound(x, 0, x_U_2, tol=tol)
        & upper_bound(x, 1, y_U_2, tol=tol)
        & upper_bound(x, 2, z_U_2, tol=tol)
    )

def mesh_interior_marker_3(x, tol=1e-12):
    return (
          lower_bound(x, 0, x_L_3, tol=tol)
        & lower_bound(x, 1, y_L_3, tol=tol)
        & lower_bound(x, 2, z_L_3, tol=tol)
        & upper_bound(x, 0, x_U_3, tol=tol)
        & upper_bound(x, 1, y_U_3, tol=tol)
        & upper_bound(x, 2, z_U_3, tol=tol)
    )

def mesh_interior_marker_4(x, tol=1e-12):
    return (
          lower_bound(x, 0, x_L_4, tol=tol)
        & lower_bound(x, 1, y_L_4, tol=tol)
        & lower_bound(x, 2, z_L_4, tol=tol)
        & upper_bound(x, 0, x_U_4, tol=tol)
        & upper_bound(x, 1, y_U_4, tol=tol)
        & upper_bound(x, 2, z_U_4, tol=tol)
    )


def main(output_path, resolution_factor):
    nx = l * 16 * 2 ** resolution_factor
    ny = 9 * 2 ** resolution_factor
    nz = 9 * 2 ** resolution_factor

    n = [nx, ny, nz]

    # Define the corners of the rectangle
    p = [np.array([0.0, 0.0, 0.0]), np.array([l*16e-6, 0.9e-6, 0.9e-6])]

    # Create the mesh
    mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD, p, n, dolfinx.mesh.CellType.hexahedron
    )

    # get topological dimension
    tdim = mesh.topology.dim

    # Get total number of cells
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts

    # Mark all cells with exterior marker
    exterior_marker = 0
    cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)

    cell_tag_1 = 1
    cell_tag_2 = 1
    cell_tag_3 = 1
    cell_tag_4 = 1

    # Remark interior cells with interior marker
    interior_cells_1 = dolfinx.mesh.locate_entities(
        mesh, tdim, mesh_interior_marker_1
    )
    cell_marker[interior_cells_1] = cell_tag_1

    # Remark interior cells with interior marker
    interior_cells_2 = dolfinx.mesh.locate_entities(
        mesh, tdim, mesh_interior_marker_2
    )
    cell_marker[interior_cells_2] = cell_tag_2

    # Remark interior cells with interior marker
    interior_cells_3 = dolfinx.mesh.locate_entities(
        mesh, tdim, mesh_interior_marker_3
    )
    cell_marker[interior_cells_3] = cell_tag_3

    # Remark interior cells with interior marker
    interior_cells_4 = dolfinx.mesh.locate_entities(
        mesh, tdim, mesh_interior_marker_4
    )
    cell_marker[interior_cells_4] = cell_tag_4

    print(max(cell_marker))
    print(min(cell_marker))

    # Create mesh tag function
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    )
    ct.name = "cell_marker"

    # Create connectivity
    mesh.topology.create_connectivity(tdim - 1, tdim)

    # Get total number of facets
    facet_map = mesh.topology.index_map(tdim - 1)
    num_facets_local = facet_map.size_local + facet_map.num_ghosts

    # Mark all facets with interior facets marker (needed for the DG method)
    interior_facet_marker = 0
    marker = np.full(num_facets_local, interior_facet_marker, dtype=np.int32)

    # Get interface facets (cell membrane facets)
    interface_facets_1 = scifem.find_interface(ct, cell_tag_1, exterior_marker)
    interface_facets_2 = scifem.find_interface(ct, cell_tag_2, exterior_marker)
    interface_facets_3 = scifem.find_interface(ct, cell_tag_3, exterior_marker)
    interface_facets_4 = scifem.find_interface(ct, cell_tag_4, exterior_marker)

    # Mark interface facets with interface marker
    marker[interface_facets_1] = cell_tag_1
    marker[interface_facets_2] = cell_tag_2
    marker[interface_facets_3] = cell_tag_3
    marker[interface_facets_4] = cell_tag_4

    # Get exterior boundary facets
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    # Mark exterior boundary facets with boundary marker
    boundary_marker = 5
    marker[exterior_facets] = boundary_marker

    # Create facet markers
    ft = dolfinx.mesh.meshtags(
        mesh, tdim - 1, np.arange(num_facets_local, dtype=np.int32), marker
    )
    ft.name = "facet_marker"

    # Define the output filename
    xdmf_filename = f"{output_path}mesh_{resolution_factor}.xdmf"

    # Create an XDMFFile object in write mode ('w')
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf_filename, "w") as xdmf:
        # Write the mesh and cell tags to the XDMF file
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)

    xdmf.close()

if __name__=="__main__":
    main("meshes/3D/", 0)
