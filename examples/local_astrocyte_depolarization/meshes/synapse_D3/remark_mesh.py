import os
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile

input_filename = "meshes/mesh.xdmf"
output_filename = "meshes/remarked_mesh.xdmf"

# Load mesh and cell tags
with XDMFFile(MPI.COMM_WORLD, input_filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cell_markers = xdmf.read_meshtags(mesh, name="Grid", attribute_name="marker")
    cell_markers = xdmf.read_meshtags(mesh, name="Grid")
    cell_labels = xdmf.read_meshtags(mesh, name="Grid", attribute_name="label")

# Retag cell markers (from previous step)
def remap_tags(old_meshtags, mesh_object):
    old_values = old_meshtags.values
    indices = old_meshtags.indices
    new_values = np.ones_like(old_values, dtype=np.int32)
    # ECS
    new_values[old_values == 1] = 0
    # Astrocyte
    new_values[old_values == 2] = 2
    # Pre and post synaptic neurons
    #new_values[old_values == 34] = 3
    #new_values[old_values == 48] = 4

    dim = mesh_object.topology.dim
    return dolfinx.mesh.meshtags(mesh_object, dim, indices, new_values)

new_cell_markers = remap_tags(cell_markers, mesh)
new_cell_labels = remap_tags(cell_labels, mesh)

# name them uniquely
new_cell_markers.name = "cell_marker"
new_cell_labels.name = "cell_label"

# generate facet markers
print("Computing facet connectivity and generating surface markers...")

tdim = mesh.topology.dim      # 3D (Cells)
fdim = tdim - 1               # 2D (Facets/Surfaces)

# Force DOLFINx to calculate facet-to-cell connectivity maps
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)

# Get the total number of local facets owned by this processor
num_facets = mesh.topology.index_map(fdim).size_local
facet_indices = np.arange(num_facets, dtype=np.int32)

# Build a fast lookup array for cell markers indexed by Cell ID
num_cells_total = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
cell_marker_lookup = np.zeros(num_cells_total, dtype=np.int32)
cell_marker_lookup[new_cell_markers.indices] = new_cell_markers.values

# Extract raw connectivity data arrays
offsets = f_to_c.offsets
connectivity_array = f_to_c.array

# Calculate how many cells are attached to each facet (1 = exterior, 2 = interior)
num_cells_per_facet = np.diff(offsets[:num_facets + 1])

# Rule: Initialize ALL facets to 0 (covers "all other interior facets with 0")
facet_values = np.zeros(num_facets, dtype=np.int32)

# Rule: All exterior facets (only 1 neighboring cell) get marked with 5
exterior_mask = (num_cells_per_facet == 1)
facet_values[exterior_mask] = 5

# Isolate interior facets (exactly 2 neighboring cells) to evaluate internal boundaries
interior_mask = (num_cells_per_facet == 2)
interior_facet_indices = np.where(interior_mask)[0]

# Grab the Cell ID for both neighbors sharing each interior facet
cell_neighbor_0 = connectivity_array[offsets[interior_facet_indices]]
cell_neighbor_1 = connectivity_array[offsets[interior_facet_indices] + 1]

# Look up what markers those neighboring cells hold
m0 = cell_marker_lookup[cell_neighbor_0]
m1 = cell_marker_lookup[cell_neighbor_1]

# Rule: Facets between cell 2 and cell 0 -> mark with 2
mask_2_0 = ((m0 == 2) & (m1 == 0)) | ((m0 == 0) & (m1 == 2))
facet_values[interior_facet_indices[mask_2_0]] = 2

# Rule: Facets between cell 1 and cell 0 -> mark with 1
mask_1_0 = ((m0 == 1) & (m1 == 0)) | ((m0 == 0) & (m1 == 1))
facet_values[interior_facet_indices[mask_1_0]] = 1

# Pack everything into a brand new DOLFINx MeshTags object
facet_markers = dolfinx.mesh.meshtags(mesh, fdim, facet_indices, facet_values)
facet_markers.name = "facet_marker"

# Convert mesh from nm to cm
mesh.geometry.x[:] *= 1e-7

# Create an XDMFFile object in write mode ('w')
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, output_filename, "w") as xdmf:
    # Write the mesh and cell tags to the XDMF file
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(new_cell_markers, mesh.geometry)
    xdmf.write_meshtags(facet_markers, mesh.geometry)

    xdmf.close()
