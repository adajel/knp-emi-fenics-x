import matplotlib as mpl
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

"""
t = 0.0  # Start time (ms)
#T = 0.1  # Final time (ms)
#num_steps = 4

#T = 0.1        # Final time (ms)
#T = 1.0e-3      # Final time (ms)
#num_steps = 2
#dt = T / num_steps  # time step size

# If choosing Option B (Fo = 0.5):
dt = 0.005       # Time step size (ms)
num_steps = 20  # Increase steps to see a nice progression over time
T = num_steps * dt  # Final time (ms)

L = 5e-4       # cm
D_K = 1.98e-9  # cm^2/ms
sigma = 5.0e-5 # standard deviation cm
"""

# Scales units
t = 0.0         # Start time (ms)
dt = 0.005      # Stable time step size (ms)
T = 0.05

num_steps = int(T/dt)
print(num_steps)

box_L = 5.0    # 5e-4 cm scaled to 5.0 um
D_K = 0.5      # 5.0e-9 cm^2/ms scaled to 0.5 um^2/ms
sigma = 0.8    # 8.0e-5 cm scaled to 0.8 um
x_c, y_c, z_c = 2.5, 2.5, 2.5

nx, ny, nz = 50, 50, 50
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([box_L, box_L, box_L])],
    [nx, ny, nz],
    mesh.CellType.tetrahedron,
)

V = fem.functionspace(domain, ("Lagrange", 1))

# Shifted center coordinates: 2.5 um
def initial_condition(x, a=5, sigma=sigma):
    return a * np.exp(-a * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2) / (2 * sigma * sigma))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(
    PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
)

xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * D_K * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Define (mean squared displacement) observables
x_coord = ufl.SpatialCoordinate(domain)
r_sq = (x_coord[0] - x_c)**2 + (x_coord[1] - y_c)**2 + (x_coord[2] - z_c)**2

# Forms to calculate total mass and variance over the domain
mass_form = fem.form(u_n * ufl.dx)
msd_form = fem.form(r_sq * u_n * ufl.dx)

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
    total_mass = fem.assemble_scalar(mass_form)
    raw_msd = fem.assemble_scalar(msd_form)

    normalized_msd = raw_msd / total_mass

    time_list.append(t)
    msd_list.append(normalized_msd)

xdmf.close()

A.destroy()
b.destroy()
solver.destroy()

time_list = np.array(time_list)
msd_list = np.array(msd_list)

slope, intercept = np.polyfit(time_list, msd_list, 1)
D_eff = slope / 6.0

lmda = np.sqrt(D_K/D_eff)

print("\n" + "="*30)
print(f"True Input D:       {D_K:e}")
print(f"Calculated D_eff:   {D_eff:e}")
print(f"Relative Error:     {abs(D_eff - D_K)/D_K * 100:.2f}%")
print(f"Tortuosity:         {lmda}")
print("="*30)
