import matplotlib as mpl
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import scifem

from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

comm = MPI.COMM_WORLD

def read_mesh(mesh_file):

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
        # Read mesh and cell tags
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode)
        ct = xdmf.read_meshtags(mesh, name='cell_marker')

        # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
        mesh.topology.create_entities(mesh.topology.dim-1)
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

        # Read facets
        ft = xdmf.read_meshtags(mesh, name='facet_marker')

    xdmf.close()

    return mesh, ct, ft

mesh_file = "meshes/remarked_mesh/mesh.xdmf"
mesh, ct, ft = read_mesh(mesh_file)

# Convert mesh from cm to um
mesh.geometry.x[:] *= 1e4

ECS = {"name":"ECS", "tag":0}

domain, _, _, _, _ = scifem.extract_submesh(mesh, ct, ECS['tag'])

"""
t = 0.0  # Start time (ms)
T = 0.5  # Final time (ms)
num_steps = 2
#sigma = 5e-5 # standard deviation cm

dt = T / num_steps  # time step size

D_K = 1.98e-8  # cm^2/ms
#sigma = 1.0e-4 # standard deviation cm
#sigma = 0.5e-4 # standard deviation cm
sigma = 5.0e-5 # standard deviation cm
x_c, y_c, z_c = 2500e-7, 2500e-7, 2500e-7
"""

# Scales units
t = 0.0         # Start time (ms)
#dt = 0.005     # Stable time step size (ms)
dt = 0.0001     # Stable time step size (ms)
#dt = 0.00005    # Stable time step size (ms)
T = 0.1

num_steps = int(T/dt)
print(num_steps)

D_K = 0.5       # 1.98e-9 cm^2/ms scaled to 0.198 um^2/ms
sigma = 0.8     # 8.0e-5 cm scaled to 0.8 u
x_c, y_c, z_c = 2.5, 2.5, 2.5

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

# Shifted center coordinates: 2.5 um
def initial_condition(x, a=5, sigma=sigma):
    return a * np.exp(-a * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2) / (2 * sigma * sigma))

u_n = dolfinx.fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

"""
# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets), V
)
"""

xdmf = dolfinx.io.XDMFFile(domain.comm, "diffusion_ECS.xdmf", "w")
xdmf.write_mesh(domain)

uh = dolfinx.fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * D_K * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_form = (u_n + dt * f) * v * ufl.dx

bilinear_form = dolfinx.fem.form(a)
linear_form = dolfinx.fem.form(L_form)

A = assemble_matrix(bilinear_form)#, bcs=[bc])
A.assemble()

# Using explicit template mapping to ensure compatibility across recent dolfinx releases
b = create_vector(dolfinx.fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Define (mean squared displacement) observables
x_coord = ufl.SpatialCoordinate(domain)
r_sq = (x_coord[0] - x_c)**2 + (x_coord[1] - y_c)**2 + (x_coord[2] - z_c)**2

# Forms to calculate total mass and variance over the domain
mass_form = dolfinx.fem.form(u_n * ufl.dx)
msd_form = dolfinx.fem.form(r_sq * u_n * ufl.dx)

# lists for time and mean squared displacement
time_list = []
msd_list = []

for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    """
    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    """

    # Solve system
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)

    # Calculate mean squared displacement (the unnormalized spatial variance of your
    # spreading Gaussian profile integrated across the entire 3D mesh).
    total_mass = dolfinx.fem.assemble_scalar(mass_form)
    raw_msd = dolfinx.fem.assemble_scalar(msd_form)

    normalized_msd = raw_msd / total_mass

    time_list.append(t)
    msd_list.append(normalized_msd)

xdmf.close()

A.destroy()
b.destroy()
solver.destroy()

time_list = np.array(time_list)
msd_list = np.array(msd_list)

#slope, intercept = np.polyfit(6 * time_list[1:], msd_list[1:], 1)
#D_eff = slope

slope, intercept = np.polyfit(time_list, msd_list, 1)
D_eff = slope / 6.0

# Tortuosity lambda calculation (with homogeneous micrometer units)
lmda = np.sqrt(D_K/D_eff)

print("\n" + "="*30)
print(f"True Input D:       {D_K:e}")
print(f"Calculated D_eff:   {D_eff:e}")
print(f"Relative Error:     {abs(D_eff - D_K)/D_K * 100:.2f}%")
print(f"Tortuosity:         {lmda}")
print("="*30)
