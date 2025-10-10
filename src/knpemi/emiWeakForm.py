from knpemi.utils import interface_normal, plus, minus, pcws_constant_project
import numpy as np
from dolfin import *

JUMP = lambda f, n: minus(f, n) - plus(f, n)

def create_measures(mesh, subdomains, surfaces):
    # define measures
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    return (dx, dS, ds)

def create_functions_emi(mesh, degree=1):
    # create function space for potential (phi)
    V = FunctionSpace(mesh, "DG", 1)
    # define function space of piecewise constants on interface gamma for solution to ODEs
    Q = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

    # current potential
    phi = Function(V)
    # previous membrane potential
    phi_M_prev_PDE = Function(Q)

    return phi, phi_M_prev_PDE

def initialize(ion_list, c_prev, physical_params, n_g, Q):
    """ calculate tissue conductance """

    # initialize
    kappa = 0

    # get physical parameters
    F = physical_params['F']
    R = physical_params['R']
    psi = physical_params['psi']
    temperature = physical_params['temperature']

    for idx, ion in enumerate(ion_list):
        if idx == len(ion_list) - 1:
            # get eliminated concentrations from previous global step
            c_ = ion_list[-1]['c']
        else:
            # get concentrations from previous global step
            c_ = split(c_prev)[idx]

        # calculate and set Nernst potential for current ion (+ is ECS, - is ICS)
        E = R * temperature / (F * ion['z']) * ln(plus(c_, n_g) / minus(c_, n_g))
        ion['E'] = pcws_constant_project(E, Q)

        # global kappa
        kappa += F * ion['z'] * ion['z'] * ion['D'] * psi * c_

    return kappa


def get_lhs_emi(kappa, u, v, dx, dS, hA, n, n_g, tau_emi, \
        physical_params, mem_models):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    # equation potential (drift terms)
    a = inner(kappa*grad(u), grad(v)) * dx \
      - inner(dot(avg(kappa*grad(u)), n('+')), jump(v)) * dS(0) \
      - inner(dot(avg(kappa*grad(v)), n('+')), jump(u)) * dS(0) \
      + tau_emi/avg(hA) * inner(avg(kappa)*jump(u), jump(v)) * dS(0)

    for mm in mem_models:
        # get tag
        tag = mm['ode'].tag
        # add coupling term at interface
        a += C_phi * inner(jump(u), jump(v))*dS(tag)

    return a

def get_rhs_emi(c_prev, v, dx, dS, n, n_g, ion_list, physical_params, \
        phi_M_prev_PDE, mem_models):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    # initialize
    L = 0

    for idx, ion in enumerate(ion_list):

        if idx == len(ion_list) - 1:
            # get eliminated concentrations from previous global step
            c_k_ = ion_list[-1]['c']
        else:
            # get concentrations from previous global step
            c_k_ = split(c_prev)[idx]

        # Add terms rhs (diffusive terms)
        L += - F * ion['z'] * inner((ion['D'])*grad(c_k_), grad(v)) * dx \
             + F * ion['z'] * inner(dot(avg((ion['D'])*grad(c_k_)), n('+')), jump(v)) * dS(0) \

    #if mms is None:
    # coupling condition at interface with splitting
    g_robin_emi = [phi_M_prev_PDE]*len(mem_models)
    #else:
        # original robin condition (without splitting)
        #g_robin_emi = [self.phi_M_prev_PDE - (1 / C_phi) * I for I in self.I_ch]

    for jdx, mm in enumerate(mem_models):
        # get tag
        tag = mm['ode'].tag
        # add robin condition at interface
        L += C_phi * inner(avg(g_robin_emi[jdx]), JUMP(v, n_g)) * dS(tag)

    return L

def get_preconditioner(V, mesh, a, kappa):

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
    mass = kappa*(1/Lp**2)*inner(up, vp)*dx

    B = a + mass

    return B


def emi_system(mesh, subdomains, surfaces, physical_params, ion_list, c_prev,
        phi, phi_M_prev_PDE, mem_models, degree=1):
    """ create and return EMI weak formulation """

    # create facet area and normals
    n = FacetNormal(mesh)
    n_g = interface_normal(subdomains, mesh)
    hA = CellDiameter(mesh)

    # DG penalty parameters
    gdim = mesh.geometry().dim()
    tau_emi = Constant(20*gdim*degree)

    # create measures
    dx, dS, ds = create_measures(mesh, subdomains, surfaces)

    # get function space
    V = phi.function_space()
    Q = phi_M_prev_PDE.function_space()

    # test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # get tissue conductance and set Nernst potentials
    kappa = initialize(ion_list, c_prev, physical_params, n_g, Q)

    lhs = get_lhs_emi(kappa,
                      u, v,
                      dx, dS, hA,
                      n, n_g, tau_emi,
                      physical_params, mem_models)

    rhs = get_rhs_emi(c_prev, v, dx, dS, n, n_g, \
                      ion_list, physical_params, phi_M_prev_PDE, mem_models)

    precond = get_preconditioner(V, mesh, lhs, kappa)

    return lhs, rhs, precond, n_g
