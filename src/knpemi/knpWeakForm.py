import dolfinx
import scifem
import numpy as np

from ufl import (
    inner,
    grad,
    TestFunctions,
    TrialFunctions,
    FacetNormal,
    MixedFunctionSpace,
    Measure,
    dot,
    FacetNormal,
)

i_res = "-"
e_res = "+"

def create_measures(mesh, ct, ft):
    # Get mesh and interface/membrane tags associated with the membrane models
    gamma_tags = np.unique(ft.values)

    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)
    dS = {}

    # Create gamma measures for each tag associated with a membrane model
    for tag in gamma_tags:
        #ordered_integration_data = scifem.compute_interface_data(ct, ct_g.find(tag))
        ordered_integration_data = scifem.compute_interface_data(ct, ft.find(tag))
        # Define measure
        dGamma = Measure(
            "dS",
            domain=mesh,
            subdomain_data=[(tag, ordered_integration_data.flatten())],
            subdomain_id=tag,
        )
        # Add to dictionary of gamma measures
        dS[tag] = dGamma

    return dx, dS, ds


def create_functions_knp(subdomain_list, ion_list, degree=1):

    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Dictionaries for lists (of functions / function-spaces for each ion) for
    # each subdomain. E.g. in case with subdomains 0 and 1 and ion species a
    # and b we have cs = {0:[c_a, c_b], 1:[c_a, c_b]}
    c = {}         # current solution
    c_prev = {}    # previous solution

    # For summing up lists of function-spaces in mixed function-space
    Vs_list = []

    #for tag in subdomain_tags:
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        mesh = subdomain['mesh_sub']
        # Create list of function-spaces for each concentration in the
        # subdomain tagged with tag
        V = dolfinx.fem.functionspace(mesh, ("CG", degree))
        V_list = [V.clone() for _ in range(N_ions)]

        # Create list of functions for each concentration in the
        # subdomain tagged with tag
        c_sub = [dolfinx.fem.Function(V) for V in V_list]
        # ... and for the concentrations in the previous time step
        c_prev_sub = [dolfinx.fem.Function(V) for V in V_list]

        # Name functions (convenient when writing results to file)
        for f, ion in zip(c_sub, ion_list): f.name =  f"c_{ion['name']}_{tag}"

        # Add lists to dictionaries
        c[tag] = c_sub
        c_prev[tag] = c_prev_sub

        Vs_list += V_list

        # Initialize and name function for eliminated ion species
        ion_list[-1][f'c_{tag}'] = dolfinx.fem.Function(V)
        ion_list[-1][f'c_{tag}'].name = f"c_{ion_list[-1]['name']}_{tag}"

    # Create mixed function-space
    W = MixedFunctionSpace(*Vs_list)

    return c, c_prev


def initialize_variables(ion_list, subdomain_list, mem_models, c_prev):
    """ Calculate sum of alpha_sum and total ionic current """
    alpha_sum = {}

    for subdomain in subdomain_list:
        tag = subdomain['tag']
        # Initialize sum for current tag
        alpha_sum_sub = 0

        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Update alpha sum
            alpha_sum_sub += ion['D'][tag] * ion['z'] * ion['z'] * c

        # Set alpha sum in dictionary
        alpha_sum[tag] = alpha_sum_sub

    # sum of ion specific channel currents for each membrane tag
    I_ch = [0]*len(mem_models)

    # loop though membrane models to set total ionic current
    for jdx, mm in enumerate(mem_models):
        # loop through ion species
        for key, value in mm['I_ch_k'].items():
            # update total channel current for each tag
            I_ch[jdx] += mm['I_ch_k'][key]

    return alpha_sum, I_ch


def create_lhs(u, v, phi, dx, dS, ion_list, physical_parameters, dt, splitting_scheme):
    """ setup variational form for the knp system """

    # get psi
    psi = physical_parameters['psi']
    phi_e = phi[0]
    phi_i = phi[1]

    # initialize form
    a = 0

    # loop over ions
    for idx, ion in enumerate(ion_list[:-1]):

        # get extra and intracellular trial and test functions
        u_e = u[0][idx]; v_e = v[0][idx]
        u_i = u[1][idx]; v_i = v[1][idx]

        # get valence and diffusion coefficients
        z = ion['z']
        D_e = ion['D'][0]
        D_i = ion['D'][1]

        # equation ion concentration diffusive + advective terms ECS contribution
        a += 1.0/dt * u_e * v_e * dx(0) \
           + inner(D_e * grad(u_e), grad(v_e)) * dx(0) \
           + z * psi * inner(D_e * u_e * grad(phi_e), grad(v_e)) * dx(0)

        # equation ion concentration diffusive + advective terms ICS contribution
        a += 1.0/dt * u_i * v_i * dx(1) \
           + inner(D_i * grad(u_i), grad(v_i)) * dx(1) \
           + z * psi * inner(D_i * u_i * grad(phi_i), grad(v_i)) * dx(1)

    return a


def create_rhs(v, phi, phi_M_prev_PDE_all, c_e_prev, c_i_prev, dx, dS,
        physical_parameters, ion_list, mem_models, I_ch, alpha_sum,
        dt, splitting_scheme):
    """ setup right hand side of variational form for KNP system """

    psi = physical_parameters['psi']        # combination of physical constants
    C_phi = physical_parameters['C_phi']    # physical parameters
    C_M = physical_parameters['C_M']        # membrane capacitance
    F = physical_parameters['F']            # Faraday's constant

    phi_e = phi[0]
    phi_i = phi[1]

    # initialize form
    L = 0

    for idx, ion in enumerate(ion_list[:-1]):
        # get extra and intracellular test functions
        v_e = v[0][idx]; v_i = v[1][idx]

        # get previous concentration
        c_e_ = c_e_prev[idx]
        c_i_ = c_i_prev[idx]

        # get valence and diffusion coefficients
        z = ion['z']
        D_e = ion['D'][0]
        D_i = ion['D'][1]

        # approximating time derivative extra and intracellular contribution
        L += 1.0/dt * c_e_ * v_e * dx(0)
        L += 1.0/dt * c_i_ * v_i * dx(1)

        # add src terms for ion injection in extracellular space
        L += ion['f_source'] * v_e * dx(0)

        # calculate alpha
        alpha_e = D_e * z * z * c_e_ / alpha_sum[0]
        alpha_i = D_i * z * z * c_i_ / alpha_sum[1]

        # calculate coupling coefficient
        C_e = alpha_e(e_res) * C_M / (F * z * dt)
        C_i = alpha_i(i_res) * C_M / (F * z * dt)

        # loop through each membrane model
        phi_M_prev_PDE = phi_M_prev_PDE_all[1]
        for jdx, mm in enumerate(mem_models):
            # get facet tag
            tag_mm = mm['ode'].tag

            if splitting_scheme:
                # robin condition terms with splitting
                g_robin_e = phi_M_prev_PDE \
                          - dt / (C_M * alpha_e(e_res)) * mm['I_ch_k'][ion['name']] \
                          + (dt / C_M) * I_ch[jdx]
                g_robin_i = phi_M_prev_PDE \
                          - dt / (C_M * alpha_i(i_res)) * mm['I_ch_k'][ion['name']] \
                          + (dt / C_M) * I_ch[jdx]
            else:
                # original robin condition terms (without splitting)
                g_robin_e = phi_M_prev_PDE \
                          - dt / (C_M * alpha_e(e_res)) * mm['I_ch_k'][ion['name']]
                g_robin_i = phi_M_prev_PDE \
                          - dt / (C_M * alpha_i(i_res)) * mm['I_ch_k'][ion['name']]

            # add robin coupling condition at interface
            L += - C_e * g_robin_e * v_e(e_res) * dS[tag_mm] \
                 + C_i * g_robin_i * v_i(i_res) * dS[tag_mm]

            # add coupling terms on interface
            L += C_e * inner(phi_i(i_res) - phi_e(e_res), v_e(e_res)) * dS[tag_mm] \
               - C_i * inner(phi_i(i_res) - phi_e(e_res), v_i(i_res)) * dS[tag_mm]

    return L


def get_rhs_mms(v, mem_models, ion_list, mms, phi_M_prev_PDE, dt,
        physical_parameters, c_prev, phi, dx, dS, ds, n):

    psi = physical_parameters['psi']        # combination of physical constants
    C_phi = physical_parameters['C_phi']    # physical parameters
    C_M = physical_parameters['C_M']        # membrane capacitance
    F = physical_parameters['F']            # Faraday's constant

    phi_e = phi[0]
    phi_i = phi[1]

    # initialize form
    L = 0

    for idx, ion in enumerate(ion_list[:-1]):
        # get extra and intracellular test functions
        v_e = v[0][idx]
        v_i = v[1][idx]

        # get previous concentration
        c_e_ = c_prev[0][idx]
        c_i_ = c_prev[1][idx]

        # get valence and diffusion coefficients
        z = ion['z']
        D_e = ion['D'][0]
        D_i = ion['D'][1]
        C_e = ion['C'][0]
        C_i = ion['C'][1]

        # approximating time derivative extra and intracellular contribution
        L += 1.0/dt * c_e_ * v_e * dx(0)
        L += 1.0/dt * c_i_ * v_i * dx(1)

        # get facet tag
        tag_mm = 1

        # original robin condition terms (without splitting)
        g_robin_e = ion['f_phi_m_e']
        g_robin_i = ion['f_phi_m_i']
        f_I_M = mms['f_I_M']

        # add robin coupling condition at interface
        L += C_i * g_robin_i * v_i(i_res) * dS[tag_mm] \
           - C_e * g_robin_e * v_e(e_res) * dS[tag_mm]

        # add coupling terms on interface
        L += C_e * inner(phi_i(i_res) - phi_e(e_res), v_e(e_res)) * dS[tag_mm] \
           - C_i * inner(phi_i(i_res) - phi_e(e_res), v_i(i_res)) * dS[tag_mm]

        L += inner(ion['f_k_e'], v_e) * dx(0)
        L += inner(ion['f_k_i'], v_i) * dx(1)

        # MMS specific: add neumann contribution
        L += - dot(ion['J_k_e'], n) * v_e * ds(5)

    return L

def knp_system(mesh, ct, ft, physical_parameters, ion_list, subdomain_list, mem_models,
        phi, phi_M_prev_PDE, c, c_prev, dt, degree=1, splitting_scheme=True,
        mms=None):
    """ Create and return EMI weak formulation """

    MMS_FLAG = False if mms is None else True

    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Create function-space for each subdomain
    V_list_total = []

    for subdomain in subdomain_list:
        tag = subdomain['tag']
        # List with function spaces of all ions in subdomain tagged with tag
        V_list = [c[tag][i].function_space for i in range(N_ions)]
        V_list_total += V_list

    # Create mixed space (for each ion in each subspace)
    W = MixedFunctionSpace(*V_list_total)

    # Create trial and test functions
    us = TrialFunctions(W)
    vs = TestFunctions(W)

    # Reorganize test and trial functions from lists into dictionary
    u = {}; v = {}
    # Get list of ions for each subdomain (ECS and cells) and insert into 
    # dictionary with cell tag as key
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        u[tag] = us[tag * N_ions:(tag + 1) * N_ions]
        v[tag] = vs[tag * N_ions:(tag + 1) * N_ions]

    # Create measures and facet normal
    dx, dS, ds = create_measures(mesh, ct, ft)
    n = FacetNormal(mesh)

    # Initialize variational formulation
    alpha_sum, I_ch = initialize_variables(
            ion_list, subdomain_list, mem_models, c_prev,
    )

    # if MMS (i.e. no ODEs to solve), set splitting_scheme to false
    if MMS_FLAG: splitting_scheme = False

    # Create left hand side knp system
    a = create_lhs(
            u, v, phi, dx, dS, ion_list, physical_parameters, dt,
            splitting_scheme
    )

    # Create right hand side knp system
    L = create_rhs(
            v, phi, phi_M_prev_PDE, c_prev[0], c_prev[1], dx, dS,
            physical_parameters, ion_list, mem_models, I_ch, alpha_sum,
            dt, splitting_scheme
    )

    # add terms specific to mms test
    if MMS_FLAG: 
        L = get_rhs_mms(
            v, mem_models, ion_list, mms, phi_M_prev_PDE, dt, physical_parameters,
            c_prev, phi, dx, dS, ds, n
        )

        return a, L, dx

    return a, L
