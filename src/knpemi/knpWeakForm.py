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
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

def create_measures(meshes, ct, ft, ct_g):
    # Get mesh and interface/membrane tags associated with the membrane models
    mesh = meshes['mesh']
    gamma_tags = np.unique(ct_g.values)

    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)
    dS = {}

    # Create gamma measures for each tag associated with a membrane model
    for tag in gamma_tags:
        ordered_integration_data = scifem.compute_interface_data(ct, ct_g.find(tag))
        # Define measure
        dGamma = Measure(
            "dS",
            domain=mesh,
            subdomain_data=[(tag, ordered_integration_data.flatten())],
            subdomain_id=tag,
        )
        # Add to dictionary of gamma measures
        dS[tag] = dGamma

    return dx, ds, dS

def create_functions_knp(meshes, ion_list, degree=1):

    mesh_e = meshes['mesh_e']
    mesh_i = meshes['mesh_i']

    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Create mixed space for extra and intracellular concentrations
    V_e = dolfinx.fem.functionspace(mesh_e, ("CG", degree))
    V_i = dolfinx.fem.functionspace(mesh_i, ("CG", degree))
    W = MixedFunctionSpace(* ([V_e] * N_ions + [V_i] * N_ions))

    # Functions for current extra and intracellular concentrations
    c_e = [dolfinx.fem.Function(V_e)] * N_ions
    c_i = [dolfinx.fem.Function(V_i)] * N_ions
    c = {'e':c_e, 'i':c_i}

    # Name functions (convenient when writing results to file)
    for f_e, f_i, ion in zip(c_e, c_i, ion_list):
        ion_name = ion['name']
        # Assign names
        f_e.name = f"c_{ion_name}_e"
        f_i.name = f"c_{ion_name}_i"

    # Functions for previous extra and intracellular concentrations
    c_e_prev = [dolfinx.fem.Function(V_e)] * N_ions
    c_i_prev = [dolfinx.fem.Function(V_i)] * N_ions
    c_prev = {'e':c_e_prev, 'i':c_i_prev}

    # Initialize function for eliminated ion species
    ion_list[-1]['c_e'] = dolfinx.fem.Function(V_e)
    ion_list[-1]['c_i'] = dolfinx.fem.Function(V_i)

    return c, c_prev

def initialize_varform(ion_list, mem_models, c_e_prev, c_i_prev):
    """ Calculate sum of alpha_sum and total ionic current """
    alpha_e_sum = 0
    alpha_i_sum = 0

    for idx, ion in enumerate(ion_list):
        if idx == len(ion_list) - 1:
            # get eliminated concentrations from previous global step
            c_e_ = ion_list[-1]['c_e']
            c_i_ = ion_list[-1]['c_i']
        else:
            # get concentrations from previous global step
            c_e_ = c_e_prev[idx]
            c_i_ = c_i_prev[idx]

        # update alpha
        alpha_e_sum += ion['D'][0] * ion['z'] * ion['z'] * c_e_
        alpha_i_sum += ion['D'][1] * ion['z'] * ion['z'] * c_i_

    # sum of ion specific channel currents for each membrane tag
    I_ch = [0]*len(mem_models)

    # loop though membrane models to set total ionic current
    for jdx, mm in enumerate(mem_models):
        # loop through ion species
        for key, value in mm['I_ch_k'].items():
            # update total channel current for each tag
            I_ch[jdx] += mm['I_ch_k'][key]

    return alpha_e_sum, alpha_i_sum, I_ch

def create_lhs_knp(u, v, phi, dx, dS, ion_list, physical_parameters, dt):
    """ setup variational form for the knp system """

    # get psi
    psi = physical_parameters['psi']
    phi_e = phi['e']
    phi_i = phi['i']

    # initialize form
    a = 0

    # loop over ions
    for idx, ion in enumerate(ion_list[:-1]):

        # get extra and intracellular trial and test functions
        u_e = u['e'][idx]; v_e = v['e'][idx]
        u_i = u['i'][idx]; v_i = v['i'][idx]

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

def create_rhs_knp(v, phi, phi_M_prev_PDE, c_e_prev, c_i_prev, dx, dS,
        physical_parameters, ion_list, mem_models, I_ch, alpha_e_sum,
        alpha_i_sum, dt):
    """ setup right hand side of variational form for KNP system """

    psi = physical_parameters['psi']        # combination of physical constants
    C_phi = physical_parameters['C_phi']    # physical parameters
    C_M = physical_parameters['C_M']        # membrane capacitance
    F = physical_parameters['F']            # Faraday's constant

    phi_e = phi['e']
    phi_i = phi['i']

    # initialize form
    L = 0

    for idx, ion in enumerate(ion_list[:-1]):
        # get extra and intracellular test functions
        v_e = v['e'][idx]; v_i = v['i'][idx]

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

        #if mms is None:
        # calculate alpha
        alpha_e = D_e * z * z * c_e_ / alpha_e_sum
        alpha_i = D_i * z * z * c_i_ / alpha_i_sum

        # calculate coupling coefficient
        C_e = alpha_e(e_res) * C_M / (F * z * dt)
        C_i = alpha_i(i_res) * C_M / (F * z * dt)

        # loop through each membrane model
        for jdx, mm in enumerate(mem_models):

            # get facet tag
            tag = mm['ode'].tag

            # robin condition with splitting
            g_robin_knp_e = phi_M_prev_PDE \
                          - dt / (C_M * alpha_e(e_res)) * mm['I_ch_k'][ion['name']] \
                          + (dt / C_M) * I_ch[jdx]

            g_robin_knp_i = phi_M_prev_PDE \
                          - dt / (C_M * alpha_i(i_res)) * mm['I_ch_k'][ion['name']] \
                          + (dt / C_M) * I_ch[jdx]
 
            #else:
            # original robin condition (without splitting)
            #g_robin_knp = self.phi_M_prev_PDE \
            #            - self.dt / (C_M * alpha) * mm['I_ch_k'][ion['name']]

            # add coupling condition at interface
            L -= C_e * g_robin_knp_e * v_e(e_res) * dS[tag]
            L += C_i * g_robin_knp_i * v_i(i_res) * dS[tag]

            # add coupling terms on interface gamma
            L -= C_i * inner(phi_i(i_res), v_i(i_res)) * dS[tag]
            L += C_i * inner(phi_e(e_res), v_i(i_res)) * dS[tag]
            L -= C_e * inner(phi_i(i_res), v_e(e_res)) * dS[tag]
            L += C_e * inner(phi_e(e_res), v_e(e_res)) * dS[tag]

    return L

def knp_system(
        meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
        phi, phi_M_prev_PDE, c, c_prev, dt, degree=1
    ):

    # Extract functions for solution concentrations
    c_e = c['e']
    c_i = c['i']

    # Extract functions for previous concentrations
    c_e_prev = c_prev['e']
    c_i_prev = c_prev['i']

    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Create measures
    dx, ds, dS = create_measures(
            meshes, ct, ft, ct_g
    )

    # Get extra and intracellular function-spaces
    V_e = c_e[0].function_space
    V_i = c_i[0].function_space

    # Create mixed space (for each ion in each subspace)
    W = MixedFunctionSpace(* ([V_e] * N_ions + [V_i] * N_ions))

    # Create trial and test functions
    us = TrialFunctions(W)
    vs = TestFunctions(W)

    # Order functions in extra and intracellular lists
    u_e = us[:N_ions]; u_i = us[N_ions:2 * N_ions]
    v_e = vs[:N_ions]; v_i = vs[N_ions:2 * N_ions]
    # ... and gather in dictionary
    u = {'e':u_e, 'i':u_i}
    v = {'e':v_e, 'i':v_i}

    # Initialize variational formulation
    alpha_e_sum, alpha_i_sum, I_ch = initialize_varform(
            ion_list, mem_models, c_e_prev, c_i_prev
    )

    # Create left hand side knp system
    a = create_lhs_knp(
            u, v, phi, dx, dS, ion_list, physical_parameters, dt
    )

    # Create right hand side knp system
    L = create_rhs_knp(
            v, phi, phi_M_prev_PDE, c_e_prev, c_i_prev, dx, dS,
            physical_parameters, ion_list, mem_models, I_ch, alpha_e_sum,
            alpha_i_sum, dt
    )

    return a, L
