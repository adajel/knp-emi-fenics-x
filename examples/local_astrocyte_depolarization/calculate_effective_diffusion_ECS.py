import matplotlib.pyplot as plt
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

colors = ['#B30000', '#E34A33', '#FC8D59', '#FDBB84']

comm = MPI.COMM_WORLD

# Define your box size (using your rescaled L = 5.0 um)
box_L = 5.0
atol = 1e-6  # Absolute tolerance to catch boundary nodes smoothly

# Define the (geometric) outer boundary
def cube_boundary_locator(x):
    on_x0 = np.isclose(x[0], 0.0, atol=atol)
    on_xL = np.isclose(x[0], box_L,   atol=atol)
    on_y0 = np.isclose(x[1], 0.0, atol=atol)
    on_yL = np.isclose(x[1], box_L,   atol=atol)
    on_z0 = np.isclose(x[2], 0.0, atol=atol)
    on_zL = np.isclose(x[2], box_L,   atol=atol)

    # a facet is on the cube boundary if it satisfies any of these conditions
    return on_x0 | on_xL | on_y0 | on_yL | on_z0 | on_zL

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

"""
#-------------------------------------------
# Cube mesh of open space with no cells (no tortuosity) to test that code and
# parameters make sense.
nx, ny, nz = 50, 50, 50
domain = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([box_L, box_L, box_L])],
    [nx, ny, nz],
    dolfinx.mesh.CellType.tetrahedron,
)
#-------------------------------------------

"""
#-------------------------------------------
# Realistic cube mesh of ECS subdomian to calculate tortuosity
mesh_file = "meshes/remarked_mesh/mesh.xdmf"
mesh, ct, ft = read_mesh(mesh_file)

# Convert mesh from cm to um
mesh.geometry.x[:] *= 1e4

ECS = {"name":"ECS", "tag":0}

domain, _, _, _, _ = scifem.extract_submesh(mesh, ct, ECS['tag'])
#-------------------------------------------

# Scaled units
t = 0.0        # Start time (ms)
dt = 0.005     # Stable time step size (ms)
T = 0.1        # End time (ms)

num_steps = int(T/dt)
print(num_steps)

D_K = 1.0                     # 1e-8 cm^2/ms scaled to 1.0 um^2/ms
sigma = 0.8                   # 8.0e-5 cm scaled to 0.8 um
x_c, y_c, z_c = 2.5, 2.5, 2.5 # mid point of mesh

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

# Shifted center coordinates to 2.5 um
def initial_condition(x, a=5, sigma=sigma):
    return a * np.exp(-a * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2) / (2 * sigma * sigma))

u_n = dolfinx.fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Get the indices of all exterior facets (tagged 1100)
fdim = domain.topology.dim - 1

# Locate the entity indices of the outer boundary facets
outer_boundary_facets = dolfinx.mesh.locate_entities_boundary(
    domain, fdim, cube_boundary_locator
)
# Dirichlet condition on outer boundary facets
bc = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(V, fdim, outer_boundary_facets), V
)

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

A = assemble_matrix(bilinear_form, bcs=[bc])
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

# Prep for plotting gaussian curves along 1D line through center of domain
num_sampling_points = 200
y_line = np.linspace(0.0, box_L, num_sampling_points)

# Create an array of 3D coordinates lying along the exact center axis
sampling_points = np.zeros((num_sampling_points, 3))
sampling_points[:, 0] = x_c
sampling_points[:, 1] = y_line
sampling_points[:, 2] = z_c

bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, sampling_points)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, sampling_points)

# Filter out points that fall inside neurons or outside boundaries
valid_indices = []
valid_points = []
cells = []

for i in range(num_sampling_points):
    links = colliding_cells.links(i)
    if len(links) > 0:  # Only append if a valid ECS cell is found!
        valid_indices.append(i)
        valid_points.append(sampling_points[i])
        cells.append(links[0])

valid_points = np.array(valid_points)
cells = np.array(cells, dtype=np.int32)

plot_steps = [0, 5, 10, 20]
profiles = {}

# Capture initial profile at t = 0 using placeholder arrays
if 0 in plot_steps:
    profiles[0] = np.full(num_sampling_points, np.nan)
    if len(cells) > 0:
        profiles[0][valid_indices] = u_n.eval(valid_points, cells).flatten()

for step in range(1, num_steps + 1):

    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

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

    # Capture 1D concentration profiles
    if step in plot_steps:
        profiles[step] = np.full(num_sampling_points, np.nan)
        if len(cells) > 0:
            profiles[step][valid_indices] = uh.eval(valid_points, cells).flatten()

xdmf.close()

A.destroy()
b.destroy()
solver.destroy()

time_list = np.array(time_list)
msd_list = np.array(msd_list)

# Find effective diffusion
slope, intercept = np.polyfit(time_list, msd_list, 1)
D_eff = slope / 6.0

# Calculate tortuosity
lmda = np.sqrt(D_K/D_eff)

print("\n" + "="*30)
print(f"True Input D:       {D_K:e}")
print(f"Calculated D_eff:   {D_eff:e}")
print(f"Relative Error:     {abs(D_eff - D_K)/D_K * 100:.2f}%")
print(f"Tortuosity:         {lmda}")
print("\n" + "="*30)

# Plot 1D concentration profiles
fig1, ax1 = plt.subplots(figsize=(7*0.7, 5*0.7))

times = [r"$\rm t=t_0$", r"$\rm t=t_1$", r"$\rm t=t_2$", r"$\rm t=t_3$"]

i = 0
for step in plot_steps:
    current_time = step * dt
    ax1.plot(y_line, profiles[step], label=times[i],
            lw=4, color=colors[i])
    # Print time index and actual time
    print(f"{times[i]}: $t = {current_time:.3f}$ ms")
    i += 1

ax1.set_xlabel(r"$\rm x$ ($\mu$m)", fontsize=11)
ax1.set_ylabel(r"$\rm c_e$ (mM)", fontsize=11)
ax1.legend(loc="upper right", frameon=True)
plt.savefig("diffusion_gaussian_profiles.svg", dpi=300, bbox_inches="tight")

# Plot mean squared displacement vs time
fig2, ax2 = plt.subplots(figsize=(7*0.7, 5*0.7))

fit_line = slope * time_list + intercept
ax2.plot(time_list, fit_line, '-', label=f"Linear fit", color="#d62728", lw=2)
ax2.plot(time_list, msd_list, 'o', label="Simulation data", color="#1f77b4", markersize=6, alpha=0.8)

ax2.set_xlabel(r"$\rm t$ (ms)", fontsize=11)
ax2.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper left", frameon=True)
plt.savefig("diffusion_msd.svg", dpi=300, bbox_inches="tight")
