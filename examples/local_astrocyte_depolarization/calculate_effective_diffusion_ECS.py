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

def create_measures(mesh, ct, ft):
    """ Create measures, all measure defined on parent mesh """
    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    # Get interface/membrane tags
    gamma_tags = np.unique(ft.values)

    subdomain_data = []
    # Define measures on membrane interface gamma
    for tag in gamma_tags:
        ordered_integration_data = scifem.compute_interface_data(ct, ft.find(tag))
        # Define measure for tag
        subdomain_data.append((tag, ordered_integration_data.flatten()))

    # Define measures on facet
    dS = Measure(
            "dS",
            domain=mesh,
            subdomain_data=subdomain_data,
        )

    return dx, dS, ds

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

ECS = {"name":"ECS",
       "tag":0,              # NB! ECS tag must always be zero.
}

neuron = {"name":"neuron",
          "tag":1,
          "membrane_tags":[1],
}

domain_0, sub_to_parent_0, sub_vertex_to_parent_0, _, _ = scifem.extract_submesh(mesh, ct, ECS['tag'])
mesh_sub_1, sub_to_parent_1, sub_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ct, neuron['tag'])
mesh_mem_1, mem_to_parent_1, mem_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ft, neuron['membrane_tags'])

t = 0.0  # Start time (ms)
T = 0.5  # Final time (ms)
num_steps = 2

#T = 0.1  # Final time (ms)
#num_steps = 4
#sigma = 5e-5 # standard deviation cm

dt = T / num_steps  # time step size

D_K = 1.98e-8  # cm^2/ms
sigma = 1.0e-4 # standard deviation cm

V0 = dolfinx.fem.functionspace(domain_0, ("Lagrange", 1))

# Shifted center coordinates: 2500e-7 = 2.5e-4 cm
def initial_condition(x, a=5, sigma=sigma):
    x_c, y_c, z_c = 2500e-7, 2500e-7, 2500e-7
    return a * np.exp(-a * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2) / (2 * sigma * sigma))

u_n = dolfinx.fem.Function(V0)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain_0.topology.dim - 1
boundary_facets = dolfinx.mesh.locate_entities_boundary(
    domain_0, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(V0, fdim, boundary_facets), V0
)

xdmf = dolfinx.io.XDMFFile(domain_0.comm, "diffusion_ECS.xdmf", "w")
xdmf.write_mesh(domain_0)

uh = dolfinx.fem.Function(V0)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

u, v = ufl.TrialFunction(V0), ufl.TestFunction(V0)
f = dolfinx.fem.Constant(domain_0, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * D_K * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx
bilinear_form = dolfinx.fem.form(a)
linear_form = dolfinx.fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(dolfinx.fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain_0.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Define (mean squared displacement) observables
x_coord = ufl.SpatialCoordinate(domain_0)
r_sq = x_coord[0]**2 + x_coord[1]**2 + x_coord[2]**2
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

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
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

slope, intercept = np.polyfit(6 * time_list[1:], msd_list[1:], 1)
D_eff = slope

print("\n" + "="*30)
print(f"True Input D:       {D_K:e}")
print(f"Calculated D_eff:   {D_eff:e}")
print(f"Relative Error:     {abs(D_eff - D_K)/D_K * 100:.2f}%")
print("="*30)
