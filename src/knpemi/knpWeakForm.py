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
    """ Create measures, all measure defined on parent mesh """
    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    # Get interface/membrane tags
    gamma_tags = np.unique(ft.values)

    # Define measures on membrane interface gamma
    dS = {}
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


def create_functions_knp(subdomain_list, ion_list, degree=1):
    """ Create functions for KNP problem. Return dictionaries c and c_prev with
        lists of respectively current and previous local concentrations for
        each subdomain. E.g. in the case with subdomains 0 and 1 and ion
        species a and b we have c = {0:[c_a, c_b], 1:[c_a, c_b]} """
    # Number of ions to solve for
    N_ions = len(ion_list[:-1])
    # Current and previous concentrations
    c = {}
    c_prev = {}

    for tag, subdomain in subdomain_list.items():
        mesh = subdomain['mesh_sub']
        # List of functionspaces for each ion in the current subdomain
        V = dolfinx.fem.functionspace(mesh, ("CG", degree))
        V_list = [V.clone() for _ in range(N_ions)]

        # Create list of functions for each concentration in the subdomain
        c_sub = [dolfinx.fem.Function(V) for V in V_list]
        # ... and list for the concentrations in the previous time step
        c_prev_sub = [dolfinx.fem.Function(V) for V in V_list]

        # Name functions (convenient when writing results to file)
        for f, ion in zip(c_sub, ion_list): f.name =  f"c_{ion['name']}_{tag}"

        # Add lists to dictionaries and to flat list
        c[tag] = c_sub
        c_prev[tag] = c_prev_sub

        # Initialize and name function for eliminated ion species
        ion_list[-1][f'c_{tag}'] = dolfinx.fem.Function(V)
        ion_list[-1][f'c_{tag}'].name = f"c_{ion_list[-1]['name']}_{tag}"

    return c, c_prev


def initialize_variables(ion_list, subdomain_list, c_prev):
    """ Calculate sum of alphas for each subdomain, and total ionic current
        I_ch for each cell (all subdomains but ECS) and each membrane model """
    # Calculate sum of alpha for each subdomain
    alpha_sum = {}
    for tag, subdomain in subdomain_list.items():
        # Initialize sum for current tag
        alpha_sum_tag = 0

        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c_tag = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]
            # Update alpha sum
            alpha_sum_tag += ion['D'][tag] * ion['z'] * ion['z'] * c_tag

        # Set alpha sum in dictionary
        alpha_sum[tag] = alpha_sum_tag

    # Calculate sum of ion specific channel currents for each cell (all
    # subdomain but the ECS) and each membrane model
    I_ch = {}
    for tag, subdomain in subdomain_list.items():
        if tag > 0:
            # Get list of membrane models for the current subdomain
            mem_models = subdomain['mem_models']

            # Calculate the total ionic channel current (I_ch_tag) for each membrane model
            # I_ch_tag[jdx] = sum of all 'I_ch_k' values (ion species currents)
            I_ch_tag = [
                sum(mm['I_ch_k'].values())
                for mm in mem_models
            ]

            # Set I_ch sum in dictionary
            I_ch[tag] = I_ch_tag

    return alpha_sum, I_ch


def create_lhs(us, vs, phi, dx, ion_list, subdomain_list, physical_params, dt):
    """ Setup left hand side of variational form for the KNP system by adding
        up the contributions from each ion species in each subdomain """
    psi = physical_params['psi']
    # Initialize form
    a = 0
    # Add contribution from each subdomain and each ion species
    for tag, subdomain in subdomain_list.items():
        for idx, ion in enumerate(ion_list[:-1]):
            # Get trial and test functions
            u = us[tag][idx]; v = vs[tag][idx]
            # Get diffusion coefficient and valence of ion
            D = ion['D'][tag]; z = ion['z']

            # Bulk dynamics equation for ion concentration (diffusive +
            # advective terms) for each ion in each subdomain
            a += 1.0/dt * u * v * dx(tag) \
               + inner(D * grad(u), grad(v)) * dx(tag) \
               + z * psi * inner(D * u * grad(phi[tag]), grad(v)) * dx(tag)

    return a


def create_rhs(vs, phi, phi_M_prev, c_prev, dx, dS, physical_params, ion_list,
        subdomain_list, I_ch, alpha_sum, dt, splitting_scheme):
    """ Setup right hand side of variational form for KNP system """
    C_M = physical_params['C_M']        # membrane capacitance
    F = physical_params['F']            # Faraday's constant

    # Initialize form
    L = 0
    for tag, subdomain in subdomain_list.items():
        for idx, ion in enumerate(ion_list[:-1]):
            # Shorthands
            v = vs[tag][idx]     # test function
            c = c_prev[tag][idx] # previous concentration

            # Approximating time derivative
            L += 1.0/dt * c * v * dx(tag)

            # Add src terms (ECS ion injection/removal)
            if tag == 0 and 'f_source' in ion:
                # Get value and cells where source terms will be applied
                value = ion['f_source']['value']
                injection_cells = ion['f_source']['injection_cells']

                # Create source term function
                mesh_sub = subdomain['mesh_sub']
                V = c.function_space
                injection_dofs = dolfinx.fem.locate_dofs_topological(V, mesh_sub.topology.dim, injection_cells)
                f = dolfinx.fem.Function(V)
                f.x.array[injection_dofs] = value

                # Set source term
                L += f * v * dx(tag)

            # Add membrane dynamics for each cellular subdomain (i.e. all subdomain but ECS)
            if tag > 0:
                # ECS and ICS (i.e. current subdomain) shorthands
                c_e = c_prev[0][idx]; c_i = c          # concentrations
                v_e = vs[0][idx]; v_i = v              # test functions
                D_e = ion['D'][0]; D_i = ion['D'][tag] # diffusion coefficients
                phi_e = phi[0]; phi_i = phi[tag]       # potentials
                z = ion['z']

                # Calculate alpha
                alpha_e = D_e * z * z * c_e / alpha_sum[0]
                alpha_i = D_i * z * z * c_i / alpha_sum[tag]
                # Calculate coupling coefficient
                C_e = alpha_e(e_res) * C_M / (F * z * dt)
                C_i = alpha_i(i_res) * C_M / (F * z * dt)

                # Loop through each membrane model (one cell tagged with tag
                # might have several membrane models with different membrane
                # tags tag_mm).
                mem_models = subdomain['mem_models']
                phi_M_prev_sub = phi_M_prev[tag]
                for jdx, mm in enumerate(mem_models):
                    # Get facet tag
                    tag_mm = mm['ode'].tag
                    if splitting_scheme:
                        # Robin condition terms with splitting
                        g_robin_e = phi_M_prev_sub \
                                  - dt / (C_M * alpha_e(e_res)) * mm['I_ch_k'][ion['name']] \
                                  + (dt / C_M) * I_ch[tag][jdx]
                        g_robin_i = phi_M_prev_sub \
                                  - dt / (C_M * alpha_i(i_res)) * mm['I_ch_k'][ion['name']] \
                                  + (dt / C_M) * I_ch[tag][jdx]
                    else:
                        # Original robin condition terms (without splitting)
                        g_robin_e = phi_M_prev_sub \
                                  - dt / (C_M * alpha_e(e_res)) * mm['I_ch_k'][ion['name']]
                        g_robin_i = phi_M_prev_sub\
                                  - dt / (C_M * alpha_i(i_res)) * mm['I_ch_k'][ion['name']]

                    # Add robin coupling condition at interface
                    L += - C_e * g_robin_e * v_e(e_res) * dS[tag_mm] \
                         + C_i * g_robin_i * v_i(i_res) * dS[tag_mm]

                    # Add coupling terms on interface
                    L += C_e * inner(phi_i(i_res) - phi_e(e_res), v_e(e_res)) * dS[tag_mm] \
                       - C_i * inner(phi_i(i_res) - phi_e(e_res), v_i(i_res)) * dS[tag_mm]

    return L


def get_rhs_mms(vs, ion_list, subdomain_list, mms, dt, c_prev, phi, dx, dS, ds, n):
    """ Get right hand side of variational form for KNP system for MMS test """
    # Initialize form
    L = 0
    for tag, subdomain in subdomain_list.items():
        for idx, ion in enumerate(ion_list[:-1]):
            # get test functions
            v = vs[tag][idx]
            # get previous concentration
            c = c_prev[tag][idx]
            # get valence and diffusion coefficients
            D = ion['D'][tag]; C = ion['C'][tag]

            # Approximating time derivative extra and intracellular contribution
            L += 1.0/dt * c * v * dx(tag)

            # Add source terms equation concentration
            if tag == 0: L += inner(ion['f_k_e'], v) * dx(tag)
            if tag == 1: L += inner(ion['f_k_i'], v) * dx(tag)

            # Add membrane dynamics for each cellular subdomain (i.e. all subdomain but ECS)
            if tag > 0:
                # ECS and ICS (i.e. current subdomain) shorthands
                C_e = ion['C'][0]; C_i = C             # coupling coefficients
                c_e = c_prev[0][idx]; c_i = c          # concentrations
                v_e = vs[0][idx]; v_i = v              # test functions
                D_e = ion['D'][0]; D_i = ion['D'][tag] # diffusion coefficients
                phi_e = phi[0]; phi_i = phi[tag]       # potentials
                z = ion['z']

                # Get dummy membrane models for MMS
                mem_models = subdomain['mem_models']
                for jdx, mm in enumerate(mem_models):
                    # Get facet tag
                    tag_mm = mm['ode'].tag

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

                    # MMS specific: add neumann contribution
                    L += - dot(ion['J_k_e'], n) * v_e * ds

    return L

def knp_system(mesh, ct, ft, physical_params, ion_list, subdomain_list,
        phi, phi_M_prev, c, c_prev, dt, degree=1, splitting_scheme=True,
        mms=None):
    """ Create and return EMI weak formulation """

    # Set MMS flag
    MMS_FLAG = False if mms is None else True
    # If MMS (i.e. no ODEs to solve), set splitting_scheme to false
    if MMS_FLAG: splitting_scheme = False

    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Create list with one function space for each ion in each subdomain
    V_list = []
    for tag, subdomain in subdomain_list.items():
        V_list += [c[tag][i].function_space for i in range(N_ions)]

    # Create mixed space (for each ion in each subspace)
    W = MixedFunctionSpace(*V_list)

    # Create trial and test functions
    us_ = TrialFunctions(W)
    vs_ = TestFunctions(W)

    # Reorganize test and trial functions: get one list of (trial or test)
    # functions for each ion for each subdomain (ECS and cells) and insert 
    # into dictionary with cell tag as key
    us = {}; vs = {}
    for idx, (tag, subdomain) in enumerate(subdomain_list.items()):
        us[tag] = us_[idx * N_ions:(idx + 1) * N_ions]
        vs[tag] = vs_[idx * N_ions:(idx + 1) * N_ions]

    # Create measures and facet normal
    dx, dS, ds = create_measures(mesh, ct, ft)
    n = FacetNormal(mesh)

    # Initialize variational formulation
    alpha_sum, I_ch = initialize_variables(ion_list, subdomain_list, c_prev)

    # Create left hand side knp system
    a = create_lhs(
            us, vs, phi, dx, ion_list, subdomain_list, physical_params, dt
    )

    # We take the preconditioner to be the system itself (P=A).
    p = a

    # Create right hand side knp system
    L = create_rhs(
            vs, phi, phi_M_prev, c_prev, dx, dS, physical_params, ion_list,
            subdomain_list, I_ch, alpha_sum, dt, splitting_scheme
    )

    # Add terms specific to mms test
    if MMS_FLAG:
        L = get_rhs_mms(
            vs, ion_list, subdomain_list, mms, dt, c_prev, phi, dx, dS, ds, n
        )

        return a, p, L, dx

    return a, p, L
