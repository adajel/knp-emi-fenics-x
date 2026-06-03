import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem import assemble_scalar

from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)

from ufl import ln

t = 0.0             # Start time (ms)
T = 0.1             # final time (ms)
num_steps = 4
dt = T / num_steps  # time step size

L = 5e-4        # cm
D_K = 1.96e-8   # diffusion coefficients K (cm²/ms)
sigma = 1.0e-4  # sd

nx, ny, nz = 50, 50, 50
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([-L/2, -L/2, -L/2]), np.array([L/2, L/2, L/2])],
    [nx, ny, nz],
    mesh.CellType.tetrahedron,
)
V = fem.functionspace(domain, ("Lagrange", 1))

def initial_condition(x, a=5, sigma=sigma):
    return a*np.exp(-a * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)/(2*sigma*sigma))

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

# =============================================================================
# 4. Define MSD Observables
# =============================================================================
x_coord = ufl.SpatialCoordinate(domain)
r_sq = x_coord[0]**2 + x_coord[1]**2 + x_coord[2]**2

# Forms to calculate total mass and variance over the domain
mass_form = fem.form(u_n * ufl.dx)
msd_form = fem.form(r_sq * u_n * ufl.dx)

time_list = []
msd_list = []

#time_history = []
#variance_history = []
#
#x_coord = ufl.SpatialCoordinate(domain)
#r_sq_expr = x_coord[0]**2 + x_coord[1]**2 + x_coord[2]**2

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

    # Mean Squared Displacement: the unnormalized spatial variance of your
    # spreading Gaussian profile integrated across the entire 3D mesh.
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

slope, intercept = np.polyfit(6 * time_list[1:], msd_list[1:], 1)
D_eff = slope

print("\n" + "="*30)
print(f"True Input D:       {D_K:e}")
print(f"Calculated D_eff:   {D_eff:e}")
print(f"Relative Error:     {abs(D_eff - D_K)/D_K * 100:.2f}%")
print("="*30)
