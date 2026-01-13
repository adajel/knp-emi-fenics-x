#!/usr/bin/env python3

"""
This script generates a 2D mesh consisting of a box with an embedded neuron
(a smaller box). The outer block as lengths

Lx = 120 um
Ly = 120 um

while the inner block has lengths

lx = 60 um
ly = 6 um
"""

import dolfinx
from mpi4py import MPI
import numpy as np
import scifem

a = [1e-6, 1e-6]
b = [61e-6, 3e-6]

# lower left point
x_L = a[0]; y_L = a[1]
# upper right point
x_U = b[0]; y_U = b[1]

def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol

def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol

def mesh_interior_marker(x, tol=1e-12):
    return (
        lower_bound(x, 0, x_L, tol=tol)
        & lower_bound(x, 1, y_L, tol=tol)
        & upper_bound(x, 0, x_U, tol=tol)
        & upper_bound(x, 1, y_U, tol=tol)
    )

def main(output_path, resolution_factor):
    # Define the number of cells in each direction
    nx = 31*2**resolution_factor
    ny = 2*2**resolution_factor
    n = [nx, ny]

    # Define the corners of the rectangle
    p = [np.array([0.0, 0.0]), np.array([62.0e-6, 4.0e-6])]

    # Create the mesh
    mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, p, n, dolfinx.mesh.CellType.triangle
    )

    # get topological dimension
    tdim = mesh.topology.dim

    # Get total number of cells
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts

    # Mark all cells with exterior marker
    exterior_marker = 0
    cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)

    # Remark interior cells with interior marker
    interior_cells = dolfinx.mesh.locate_entities(
        mesh, tdim, mesh_interior_marker
    )
    interior_marker = 1
    cell_marker[interior_cells] = interior_marker

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
    interface_facets = scifem.find_interface(ct, interior_marker, exterior_marker)
    # Mark interface facets with interface marker
    interface_marker = 1
    marker[interface_facets] = interface_marker

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
    fname = "2D"
    main(f"meshes/{fname}/", 0)
    main(f"meshes/{fname}/", 1)
    main(f"meshes/{fname}/", 2)
    main(f"meshes/{fname}/", 3)
