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

x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75


def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol


def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol


def omega_interior_marker(x, tol=1e-12):
    return (
        lower_bound(x, 0, x_L, tol=tol)
        & lower_bound(x, 1, y_L, tol=tol)
        & upper_bound(x, 0, x_U, tol=tol)
        & upper_bound(x, 1, y_U, tol=tol)
    )

def main(output_path, M, resolution_factor):

    #M = 10*resolution_factor
    omega = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
        )

    interior_cells = dolfinx.mesh.locate_entities(
    omega, omega.topology.dim, omega_interior_marker

    )
    interior_marker = 1
    exterior_marker = 0
    cell_map = omega.topology.index_map(omega.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)
    cell_marker[interior_cells] = interior_marker
    ct = dolfinx.mesh.meshtags(
        omega, omega.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    )
    ct.name = "cell_marker"

    gamma_facets = scifem.find_interface(ct, interior_marker, exterior_marker)

    omega.topology.create_connectivity(omega.topology.dim - 1, omega.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
    facet_map = omega.topology.index_map(omega.topology.dim - 1)
    num_facets_local = facet_map.size_local + facet_map.num_ghosts
    facets = np.arange(num_facets_local, dtype=np.int32)
    interface_marker = 1
    boundary_marker = 5
    marker = np.full_like(facets, -1, dtype=np.int32)
    marker[gamma_facets] = interface_marker
    marker[exterior_facets] = boundary_marker
    marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
    ft = dolfinx.mesh.meshtags(
        omega, omega.topology.dim - 1, marker_filter, marker[marker_filter]
    )
    ft.name = "facet_marker"

    # Define the output filename
    xdmf_filename = f"{output_path}mesh_{resolution_factor}.xdmf"

    # Create an XDMFFile object in write mode ('w')
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf_filename, "w") as xdmf:
        # Write the mesh and cell tags to the XDMF file
        xdmf.write_mesh(omega)
        xdmf.write_meshtags(ct, omega.geometry)
        xdmf.write_meshtags(ft, omega.geometry)

    xdmf.close()

if __name__=="__main__":
    main("meshes/mms/", 100, 2)
    main("meshes/mms/", 200, 3)
    main("meshes/mms/", 400, 4)
    main("meshes/mms/", 800, 5)
    main("meshes/mms/", 1600, 6)
