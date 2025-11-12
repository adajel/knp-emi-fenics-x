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
    """ Create measures, all measure defined on parent mesh """
    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    # Get interface/membrane tags
    gamma_tags = np.unique(ft.values)
    dS = {}

    # Define measures on membrane interface gamma
    for tag in gamma_tags:
        ordered_integration_data = scifem.compute_interface_data(ct, ft.find(tag))
        # Define measure for tag
        dS_tag = Measure(
                "dS",
                domain=mesh,
                subdomain_data=[(tag, ordered_integration_data.flatten())],
                subdomain_id=tag,
        )
        # Add measure to dictionary with all gamma measures
        dS[tag] = dS_tag

    return dx, dS, ds


def create_functions_emi(subdomain_list, degree=1):
    """ Create functions for EMI problem. Return dictionary phi containing
        local potentials for each subdomain and function for previous membrane
        potential phi_M_prev. E.g. in the case with subdomains 0 and 1 we have
        phi = {0:phi_e, 1:phi_i} """
    phi = {}
    for subdomain in subdomain_list:
        tag = subdomain["tag"]
        mesh = subdomain["mesh_sub"]
        # Create local functionspace for local potential ..
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


def initialize_variables(ion_list, subdomain_list, c_prev, physical_params):
    """ Calculate kappa (tissue conductance) and set Nernst potentials """
    # Get physical parameters
    F = physical_params['F']
    psi = physical_params['psi']

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

            # Calculate and set Nernst potential for cells (all subdomains but ECS)
            if tag != 0:
                # ECS concentration (ECS is subdomain with tag 0)
                c_e = ion_list[-1][f'c_0'] if is_last else c_prev[0][idx]
                # Calculate and set Nernst potential
                ion[f'E_{tag}'] = 1 / (psi * ion['z']) * ln(c_e(e_res) / c(i_res))

        # Add to dictionary
        kappa[tag] = kappa_sub

    # Calculate sum of ion specific channel currents for each cell (all
    # subdomain but the ECS) and each membrane model
    I_ch = {}
    for subdomain in subdomain_list[1:]:
        tag = subdomain['tag']
        mem_models = subdomain['mem_models']
        I_ch_tag = [0]*len(mem_models)

        # Loop though membrane models to set total ionic current
        for jdx, mm in enumerate(mem_models):
            # loop through ion species
            for key, value in mm['I_ch_k'].items():
                # update total channel current for each tag
                I_ch_tag[jdx] += mm['I_ch_k'][key]

        # Set I_ch sum in dictionary
        I_ch[tag] = I_ch_tag

    return kappa, I_ch


def get_lhs(us, vs, dx, dS, subdomain_list, physical_params, kappa, splitting_scheme):
    """ Setup left hand side of the variational form for the emi system """
    C_phi = physical_params['C_phi']
    a = 0
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        # Get test and trial functions
        u = us[tag]; v = vs[tag]

        # Add contribution of subdomain to equation for potential (drift terms)
        a += inner(kappa[tag] * grad(u), grad(v)) * dx(tag)

        # Add membrane dynamics for each cell (all subdomain but ECS)
        if tag > 0:
            # ECS and ICS (i.e. current subdomain) shorthands
            u_e = us[0]; u_i = u # trial functions
            v_e = vs[0]; v_i = v # test functions

            # Loop through each membrane model (one cell tagged with tag
            # might have several membrane models with different membrane
            # tags tag_mm).
            mem_models = subdomain['mem_models']
            for mm in mem_models:
                # Get membrane model tag
                tag_mm = mm['ode'].tag
                # add coupling term at interface
                a += C_phi * (u_i(i_res) - u_e(e_res)) * v_i(i_res) * dS[tag_mm] \
                   - C_phi * (u_i(e_res) - u_e(i_res)) * v_e(e_res) * dS[tag_mm]

    return a

def get_rhs(c_prev, vs, dx, dS, ion_list, subdomain_list, physical_params, 
        phi_M_prev, I_ch, splitting_scheme):
    """ Setup right hand side of variational form for the EMI system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    L = 0
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        v = vs[tag]
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'][tag])*grad(c), grad(v)) * dx(tag)

        # Add membrane dynamics for each cell (all subdomain but ECS)
        if tag > 0:
            # ECS and ICS (i.e. current subdomain) shorthands
            v_e = vs[0]; v_i = v # test functions

            # Loop through each membrane model (one cell tagged with tag
            # might have several membrane models with different membrane
            # tags tag_mm).
            mem_models = subdomain['mem_models']
            for jdx, mm in enumerate(mem_models):
                # Get facet tag
                tag_mm = mm['ode'].tag
                if splitting_scheme:
                    # Robin condition with PDE/ODE splitting scheme
                    g_robin = phi_M_prev[tag]
                else:
                    # Original robin condition (without splitting)
                    g_robin = phi_M_prev[tag] - (1 / C_phi) * I_ch[tag][jdx]

                # Add robin condition at interface
                L += C_phi * inner(g_robin, v_i(i_res) - v_e(e_res)) * dS[tag_mm]

    return L


def get_rhs_mms(vs, dx, dS, ds, c_prev, ion_list, subdomain_list,
        physical_params, dt, n, mms):

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    L = 0
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        v = vs[tag]
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'][tag])*grad(c), grad(v)) * dx(tag)

            # Add Neumann term (zero in physiological simulation)
            if tag == 0: L += - F * ion['z'] * dot(ion['J_k_e'], n) * v * ds(5)

        # EMI source terms for potentials
        if tag == 0: L += inner(mms['f_phi_e'], v) * dx(tag)
        if tag == 1: L += inner(mms['f_phi_i'], v) * dx(tag)

        # Add membrane dynamics for each cell (all subdomain but ECS)
        if tag > 0:
            # ECS and ICS (i.e. current subdomain) shorthands
            v_e = vs[0]; v_i = v # test functions

            # Loop through each membrane model (one cell tagged with tag
            # might have several membrane models with different membrane
            # tags tag_mm).
            mem_models = subdomain['mem_models']
            for jdx, mm in enumerate(mem_models):
                # Get facet tag
                tag_mm = mm['ode'].tag
                # Add robin terms (i.e. source term for equation for phi_M)
                L += C_phi * inner(mms['f_phi_m'], v_i(i_res) - v_e(e_res)) * dS[tag_mm]
                # Enforcing correction for I_m
                L -= inner(mms['f_I_M'], v_e(e_res)) * dS[tag_mm]

    return L

def emi_system(mesh, ct, ft, physical_params, ion_list, subdomain_list, mem_models,
        phi, phi_M_prev, c_prev, dt, degree=1, splitting_scheme=True, mms=None):
    """ Create and return EMI weak formulation """

    # Set MMS flag
    MMS_FLAG = False if mms is None else True
    # If MMS (i.e. no ODEs to solve), set splitting_scheme to false
    if MMS_FLAG: splitting_scheme = False

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
            ion_list, subdomain_list, c_prev, physical_params
    )

    # get standard variational formulation
    a = get_lhs(
            u, v, dx, dS, subdomain_list, physical_params, kappa, splitting_scheme
    )

    L = get_rhs(
            c_prev, v, dx, dS, ion_list, subdomain_list, physical_params,
            phi_M_prev, I_ch, splitting_scheme
    )

    # add terms specific to mms test
    if MMS_FLAG:
        L = get_rhs_mms(
                v, dx, dS, ds, c_prev, ion_list, subdomain_list, 
                physical_params, dt, n, mms
        )

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
