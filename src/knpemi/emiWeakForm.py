import numpy as np
import dolfinx
import scifem

from ufl import (
    inner,
    grad,
    dot,
    TestFunctions,
    TrialFunctions,
    MixedFunctionSpace,
    Measure,
    ln,
    extract_blocks,
    FacetNormal,
)

i_res = "-"
e_res = "+"

def create_measures(mesh, ct, ft):
    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    dS = {}
    gamma_tags = np.unique(ft.values)

    # Define measures on membrane interface gamma
    for tag in gamma_tags:
        ordered_integration_data = scifem.compute_interface_data(ct, ft.find(tag))
        dS_tag = Measure("dS",
                    domain=mesh,
                    subdomain_data=[(tag, ordered_integration_data.flatten())],
                    subdomain_id=tag,
                    )
        dS[tag] = dS_tag

    return dx, dS, ds

def create_functions_emi(subdomain_list, degree=1):

    phi = {}
    for subdomain in subdomain_list:
        tag = subdomain["tag"]
        mesh = subdomain["mesh_sub"]
        # Create local functions space for local potential ..
        V = dolfinx.fem.functionspace(mesh, ("CG", degree))
        # ... and create and name function for local potential.
        phi_sub = dolfinx.fem.Function(V)
        phi_sub.name = f"phi_{tag}"

        # Add local function to dictionary
        phi[tag] = phi_sub

    neuron = subdomain_list[1]
    mesh_g = neuron["mesh_mem"]
    # create function space over gamma for membrane potential (phi_M)
    Q = dolfinx.fem.functionspace(mesh_g, ("CG", degree))
    # previous membrane potential
    phi_M_prev = {1:dolfinx.fem.Function(Q)}

    return phi, phi_M_prev


def initialize_variables(ion_list, c_prev, physical_params, mem_models, subdomain_list):
    """ Calculate kappa (tissue conductance) and set Nernst potentials """
    # Get physical parameters
    F = physical_params['F']
    R = physical_params['R']
    psi = physical_params['psi']
    temperature = physical_params['temperature']

    # Initialize dictionary
    kappa = {}

    for subdomain in subdomain_list:
        tag = subdomain['tag']
        # Initialize kappa for each subdomain
        kappa_sub = 0

        # For each ion ...
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add contribution to kappa (tissue conductance)
            kappa_sub += F * ion['z'] * ion['z'] * ion['D'][tag] * psi * c

            if tag != 0:
                # Get ECS concentration (ECS is subdomain with tag 0)
                c_e = ion_list[-1][f'c_0'] if is_last else c_prev[0][idx]
                # Calculate and set Nernst potential
                ion[f'E_{tag}'] = R * temperature / (F * ion['z']) * ln(c_e(e_res) / c(i_res))

        # Add to dictionary
        kappa[tag] = kappa_sub

    # sum of ion specific channel currents for each membrane tag
    I_ch = [0]*len(mem_models)

    # loop though membrane models to set total ionic current
    for jdx, mm in enumerate(mem_models):
        # loop through ion species
        for key, value in mm['I_ch_k'].items():
            # update total channel current for each tag
            I_ch[jdx] += mm['I_ch_k'][key]

    return kappa, I_ch


def get_lhs(kappa, u, v, dx, dS, physical_params, mem_models,
        splitting_scheme):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    kappa_e = kappa[0]
    kappa_i = kappa[1]

    # get test and trial functions for each of the sub-domains
    u_e = u[0]; u_i = u[1]
    v_e = v[0]; v_i = v[1]

    # equation potential (drift terms)
    a = inner(kappa_e * grad(u_e), grad(v_e)) * dx(0) \
      + inner(kappa_i * grad(u_i), grad(v_i)) * dx(1) \

    for mm in mem_models:
        # get tag
        tag = mm['ode'].tag
        # add coupling term at interface
        a += C_phi * (u_i(i_res) - u_e(e_res)) * v_i(i_res) * dS[tag] \
           - C_phi * (u_i(e_res) - u_e(i_res)) * v_e(e_res) * dS[tag]

    return a

def get_rhs(c_prev, v, dx, dS, ion_list, physical_params, phi_M_prev,
        mem_models, I_ch, splitting_scheme):
    """ setup variational form for the emi system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    v_e = v[0]
    v_i = v[1]

    # initialize
    L = 0

    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)

        c_e_ = ion_list[-1]['c_0'] if is_last else c_prev[0][idx]
        c_i_ = ion_list[-1]['c_1'] if is_last else c_prev[1][idx]

        # Add terms rhs (diffusive terms)
        L += - F * ion['z'] * inner((ion['D'][0])*grad(c_e_), grad(v_e)) * dx(0) \
             - F * ion['z'] * inner((ion['D'][1])*grad(c_i_), grad(v_i)) * dx(1) \

    if splitting_scheme:
        # robin condition with PDE/ODE splitting scheme
        g_robin = [phi_M_prev[1]]*len(mem_models)
    else:
        # original robin condition (without splitting)
        g_robin = [phi_M_prev[1] - (1 / C_phi) * I for I in I_ch]

    for jdx, mm in enumerate(mem_models):
        # get tag
        tag = mm['ode'].tag
        # add robin condition at interface
        L += C_phi * inner(g_robin[jdx], v_i(i_res) - v_e(e_res)) * dS[tag]

    return L

def get_rhs_mms(v, dx, dS, ds, dt, n, c_prev,
        mms, physical_params, mem_models, ion_list):

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    psi = physical_params['psi']
    C_M = physical_params['C_M']
    R = physical_params['R']
    temperature = physical_params['temperature']

    v_e = v[0]
    v_i = v[1]

    # initialize
    L = 0

    # not MMS specific,
    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)

        c_e_ = ion_list[-1]['c_0'] if is_last else c_prev[0][idx]
        c_i_ = ion_list[-1]['c_1'] if is_last else c_prev[1][idx]

        # Add terms rhs (diffusive terms)
        L += - F * ion['z'] * inner((ion['D'][0])*grad(c_e_), grad(v_e)) * dx(0) \
             - F * ion['z'] * inner((ion['D'][1])*grad(c_i_), grad(v_i)) * dx(1) \

    # MMS specific source terms
    # EMI source terms for potentials
    L += inner(mms['f_phi_e'], v_e) * dx(0) # Equation for phi_e
    L += inner(mms['f_phi_i'], v_i) * dx(1) # Equation for phi_i

    # Add robin terms (i.e. source term for equation for phi_M)
    L += C_phi * inner(mms['f_phi_m'], v_i(i_res) - v_e(e_res)) * dS[1]
    # Enforcing correction for I_m
    L -= inner(mms['f_I_M'], v_e(e_res)) * dS[1]

    # Add Neumann term (zero in physiological simulation)
    for ion in ion_list:
        L += - F * ion['z'] * dot(ion['J_k_e'], n) * v_e * ds(5)

    return L

def emi_system(mesh, ct, ft, physical_params, ion_list, subdomain_list, mem_models,
        phi, phi_M_prev, c_prev, dt, degree=1, splitting_scheme=True, mms=None):
    """ Create and return EMI weak formulation """

    MMS_FLAG = False if mms is None else True

    # Create function-space for each subdomain
    Vs = []
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        V = phi[tag].function_space
        Vs.append(V)

    # Create mixed function space for potentials (phi)
    W = MixedFunctionSpace(*Vs)
    # Create trial and test functions
    us = TrialFunctions(W)
    vs = TestFunctions(W)

    u = {}; v = {}

    # Add test and trial function to dictionary with subdomain tags as keys
    for tag, u_ in enumerate(us): u[tag] = u_
    for tag, v_ in enumerate(vs): v[tag] = v_

    # Create measures and facet normal
    dx, dS, ds = create_measures(mesh, ct, ft)
    n = FacetNormal(mesh)

    # Get tissue conductance and set Nernst potentials
    kappa, I_ch = initialize_variables(
            ion_list, c_prev, physical_params, mem_models, subdomain_list,
    )

    # if MMS (i.e. no ODEs to solve), set splitting_scheme to false
    if MMS_FLAG: splitting_scheme = False

    # get standard variational formulation
    a = get_lhs(
            kappa, u, v, dx, dS, physical_params, mem_models,
            splitting_scheme
        )

    L = get_rhs(
            c_prev, v, dx, dS, ion_list, physical_params, phi_M_prev,
            mem_models, I_ch, splitting_scheme
    )

    # add terms specific to mms test
    if MMS_FLAG:
        L = get_rhs_mms(v, dx, dS, ds, dt, n, c_prev, mms, physical_params,
                mem_models, ion_list)

        # Create Dirichlet BC
        #omega_e = meshes['mesh_sub_0']
        #e_vertex_to_parent = meshes['e_vertex_to_parent']
        #exterior_to_parent = meshes['e_to_parent']
        #boundary_marker = 5
        #sub_tag, _ = scifem.transfer_meshtags_to_submesh(
        #    ft, omega_e, e_vertex_to_parent, exterior_to_parent
        #)

        #omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
        #bc_dofs = dolfinx.fem.locate_dofs_topological(
        #    V_e, omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
        #)

        #u_bc = dolfinx.fem.Function(V_e)
        #u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
        #bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)

        return a, L, dx

    return a, L
