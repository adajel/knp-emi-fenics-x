import numpy as np
import dolfinx
import scifem

from ufl import (
    inner,
    grad,
    TestFunctions,
    TrialFunctions,
    MixedFunctionSpace,
    Measure,
    ln,
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

def create_measures(meshes, ct, ft, ct_g):

    mesh = meshes['mesh']

    gamma_tags = np.unique(ct_g.values)

    # define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    dS = {}

    for tag in gamma_tags:
        ordered_integration_data = scifem.compute_interface_data(ct, ct_g.find(tag))
        dS_tag = Measure("dS", domain=mesh,
                    subdomain_data=[(tag,
                    ordered_integration_data.flatten())])
        dS[tag] = dS_tag

    return dx, dS, ds

def create_functions_emi(meshes, degree=1):

    mesh_e = meshes["mesh_e"]
    mesh_i = meshes["mesh_i"]
    mesh_g = meshes["mesh_g"]

    # create functions spaces for phi_e and phi_i over Omega_e and Omega_i respectively
    V_e = dolfinx.fem.functionspace(mesh_e, ("CG", degree))
    V_i = dolfinx.fem.functionspace(mesh_i, ("CG", degree))

    # create function space over gamma for membrane potential (phi_M)
    Q = dolfinx.fem.functionspace(mesh_g, ("CG", degree))

    # current potential
    phi_e = dolfinx.fem.Function(V_e)
    phi_i = dolfinx.fem.Function(V_i)
    # previous membrane potential
    phi_M_prev_PDE = dolfinx.fem.Function(Q)

    # name functions (convenient when writing results to file)
    phi_e.name = "phi_e"
    phi_i.name = "phi_i"
    phi_M_prev_PDE.name = "phi_m"

    phi = {'e':phi_e, 'i':phi_i}

    return phi, phi_M_prev_PDE

def initialize_varform(ion_list, c_prev, physical_params):
    """ Calculate kappa (tissue conductance) and set Nernst potentials """
    # Initialize
    kappa_e = 0; kappa_i = 0

    # Get physical parameters
    F = physical_params['F']
    R = physical_params['R']
    psi = physical_params['psi']
    temperature = physical_params['temperature']

    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)

        c_e = ion_list[-1]['c_e'] if is_last else c_prev['e'][idx]
        c_i = ion_list[-1]['c_i'] if is_last else c_prev['i'][idx]

        # Calculate and set Nernst potential for current ion (+ is ECS, - is ICS)
        ion['E'] = R * temperature / (F * ion['z']) * ln(c_i(i_res) / c_e(e_res))

        # Add contribution to kappa (tissue conductance)
        kappa_e += F * ion['z'] * ion['z'] * ion['D'][0] * psi * c_e
        kappa_i += F * ion['z'] * ion['z'] * ion['D'][1] * psi * c_i

    kappa = {'e':kappa_e, 'i':kappa_i}

    return kappa


def get_lhs_emi(kappa, u, v, dx, dS, physical_params, mem_models):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    kappa_e = kappa['e']
    kappa_i = kappa['i']

    # get test and trial functions for each of the sub-domains
    u_e = u['e']; u_i = u['i']
    v_e = v['e']; v_i = v['i']

    # equation potential (drift terms)
    a = inner(kappa_e * grad(u_e), grad(v_e)) * dx(0) \
      + inner(kappa_i * grad(u_i), grad(v_i)) * dx(1) \

    for mm in mem_models:
        # get tag
        tag = mm['ode'].tag
        # add coupling term at interface
        a += C_phi * (u_i(i_res) - u_e(e_res)) * v_i(i_res) * dS[tag] \
           + C_phi * (u_e(e_res) - u_i(i_res)) * v_e(e_res) * dS[tag]

    return a

def get_rhs_emi(c_prev, v, dx, dS, ion_list, physical_params, phi_M_prev_PDE, mem_models):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    v_e = v['e']
    v_i = v['i']

    # initialize
    L = 0

    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)

        c_e_ = ion_list[-1]['c_e'] if is_last else c_prev['e'][idx]
        c_i_ = ion_list[-1]['c_i'] if is_last else c_prev['i'][idx]

        # Add terms rhs (diffusive terms)
        L += - F * ion['z'] * inner((ion['D'][0])*grad(c_e_), grad(v_e)) * dx(0) \
             - F * ion['z'] * inner((ion['D'][1])*grad(c_i_), grad(v_i)) * dx(1) \

    # coupling condition at interface with splitting
    g_robin_emi = [phi_M_prev_PDE]*len(mem_models)
    #else:
        # original robin condition (without splitting)
        #g_robin_emi = [self.phi_M_prev_PDE - (1 / C_phi) * I for I in self.I_ch]

    for jdx, mm in enumerate(mem_models):
        # get tag
        tag = mm['ode'].tag
        # add robin condition at interface
        L += C_phi * inner(g_robin_emi[jdx], v_e(e_res)) * dS[tag] \
           - C_phi * inner(g_robin_emi[jdx], v_i(i_res)) * dS[tag]

    return L


def get_preconditioner(V, mesh, a, kappa_e, kappa_i):

   # setup preconditioner EMI
    up, vp = TrialFunction(V), TestFunction(V)

    # scale mass matrix to get condition number independent from domain length
    gdim = mesh.geometry().dim()

    for axis in range(gdim):
        x_min = mesh.coordinates().min(axis=0)
        x_max = mesh.coordinates().max(axis=0)

        x_min = np.array([MPI.min(mesh.mpi_comm(), xi) for xi in x_min])
        x_max = np.array([MPI.max(mesh.mpi_comm(), xi) for xi in x_max])

    # scaled mess matrix
    Lp = Constant(max(x_max - x_min))
    # self.B_emi is singular so we add (scaled) mass matrix
    mass = kappa_e*(1/Lp**2)*inner(up, vp)*dx

    B = a + mass

    return B


def emi_system(meshes, ct, ft, ct_g, physical_params, ion_list, mem_models,
        phi, phi_M_prev_PDE, c_prev, degree=1):
    """ Create and return EMI weak formulation """

    phi_e = phi['e']
    phi_i = phi['i']

    # Create measures
    dx, dS, ds = create_measures(meshes, ct, ft, ct_g)

    # Get function space
    V_e = phi_e.function_space
    V_i = phi_i.function_space
    # Create mixed function space for potentials (phi)
    W = MixedFunctionSpace(*[V_e, V_i])

    # Test and trial functions
    u_e, u_i = TrialFunctions(W)
    v_e, v_i = TestFunctions(W)

    u = {'e':u_e, 'i':u_i}
    v = {'e':v_e, 'i':v_i}

    # Get tissue conductance and set Nernst potentials
    kappa = initialize_varform(
            ion_list, c_prev, physical_params
    )

    a = get_lhs_emi(
            kappa, u, v, dx, dS, physical_params, mem_models
    )

    L = get_rhs_emi(
            c_prev, v, dx, dS, ion_list, physical_params, phi_M_prev_PDE, mem_models
    )

    # get function space at gamma for membrane potential
    #V = phi_M_prev_PDE.function_space
    #precond = get_preconditioner(V, mesh, lhs, kappa_e, kappa_i)

    return a, L
