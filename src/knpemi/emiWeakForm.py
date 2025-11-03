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
    extract_blocks
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

def create_measures(meshes, ct, ft, ct_g):
    mesh = meshes['mesh']
    gamma_tags = np.unique(ct_g.values)

    # Define measures
    dx = Measure('dx', domain=mesh, subdomain_data=ct)
    ds = Measure('ds', domain=mesh, subdomain_data=ft)

    dS = {}
    # Define measures on membrane interface gamma
    interface_marker = 4
    #ordered_integration_data = scifem.compute_interface_data(ct, ct_g.find(interface_marker))
    ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
    for tag in gamma_tags:
        dS_tag = Measure("dS",
                    domain=mesh,
                    subdomain_data=[(tag, ordered_integration_data.flatten())],
                    subdomain_id=tag,
                    )
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
    phi_M_prev = dolfinx.fem.Function(Q)

    # name functions (convenient when writing results to file)
    phi_e.name = "phi_e"
    phi_i.name = "phi_i"
    phi_M_prev.name = "phi_m"

    phi = {'e':phi_e, 'i':phi_i}

    return phi, phi_M_prev


def initialize_variables(ion_list, c_prev, physical_params, mem_models):
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
        ion['E'] = R * temperature / (F * ion['z']) * ln(c_e(e_res) / c_i(i_res))

        # Add contribution to kappa (tissue conductance)
        kappa_e += F * ion['z'] * ion['z'] * ion['D'][0] * psi * c_e
        kappa_i += F * ion['z'] * ion['z'] * ion['D'][1] * psi * c_i

    kappa = {'e':kappa_e, 'i':kappa_i}

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


def get_rhs(c_prev, v, dx, dS, ion_list, physical_params, phi_M_prev,

        mem_models, I_ch, splitting_scheme):
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

    if splitting_scheme:
        # robin condition with PDE/ODE splitting scheme
        g_robin = [phi_M_prev]*len(mem_models)
    else:
        # original robin condition (without splitting)
        g_robin = [phi_M_prev - (1 / C_phi) * I for I in I_ch]

    for jdx, mm in enumerate(mem_models):
        # get tag
        tag = mm['ode'].tag
        # add robin condition at interface
        L += C_phi * inner(g_robin[jdx], v_e(e_res)) * dS[tag] \
           - C_phi * inner(g_robin[jdx], v_i(i_res)) * dS[tag]

    return L

def add_mms_terms(a, L, v, dx, dS, ds, I_ch, phi_M_prev,
        mms, physical_params, mem_models, ion_list):

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    """
    #-----------------------------------------------------------------------
    # Remove robin splitting terms from variational formulation (to be replaced
    # with MMS terms below)

    # original robin condition (without splitting)
    g_robin = [phi_M_prev - (1 / C_phi) * I for I in I_ch]

    for jdx, mm in enumerate(mem_models):
        # get tag
        tag = mm['ode'].tag
        # add robin condition at interface
        L -= C_phi * inner(g_robin[jdx], v['e'](e_res)) * dS[tag] \
           - C_phi * inner(g_robin[jdx], v['i'](i_res)) * dS[tag]
    #-----------------------------------------------------------------------
    """

    f_flux = mms['f_flux_phi']
    g_robin = mms['g_robin_phi']

    L = 0

    # Add MMS term for g_robin
    # ... add robin condition at interface
    for mm in mem_models:
        # get tag
        tag = mm['ode'].tag
        # add robin MMS condition at interface
        #L += C_phi * inner(g_robin, v['i'](i_res)) * dS[tag] \
           #- C_phi * inner(g_robin, v['e'](e_res)) * dS[tag]

        print("tag mem model", tag)

        # MMS specific: we don't have normal cont. of I_M across interface
        #L += inner(f_flux, v['e'](e_res)) * dS[tag]

    f_phi_e = mms['f_phi_e']
    f_phi_i = mms['f_phi_i']
    n = mms['n']

    # MMS specific: add MMS source terms to bulk
    L += f_phi_e * v['e'] * dx(0) \
       + f_phi_i * v['i'] * dx(1)

    # MMS specific: add neumann boundary conditions
    for ion in ion_list:
        # MMS specific: add neumann boundary terms (not zero in MMS case)
        L += - F * ion['z'] * dot(ion['bdry'], n) * v['e'] * ds

    #g = mms['neumann_emi']
    #L += inner(g, v['e']) * ds

    return a, L


def emi_system(meshes, ct, ft, ct_g, physical_params, ion_list, mem_models,
        phi, phi_M_prev, c_prev, degree=1, splitting_scheme=True, mms=None):
    """ Create and return EMI weak formulation """

    MMS_FLAG = False if mms is None else True

    phi_e = phi['e']
    phi_i = phi['i']

    # Create measures
    dx, dS, ds = create_measures(meshes, ct, ft, ct_g)

    # Get function space
    V_e = phi_e.function_space
    V_i = phi_i.function_space
    # Create mixed function space for potentials (phi)
    W = MixedFunctionSpace(V_e, V_i)

    # Test and trial functions
    ue, ui = TrialFunctions(W)
    ve, vi = TestFunctions(W)

    u = {'e':ue, 'i':ui}
    v = {'e':ve, 'i':vi}

    # Get tissue conductance and set Nernst potentials
    kappa, I_ch = initialize_variables(
            ion_list, c_prev, physical_params, mem_models
    )

    # TODO
    #kappa_e = dolfinx.fem.Constant(meshes['mesh_e'], 1.0)
    #kappa_i = dolfinx.fem.Constant(meshes['mesh_i'], 2.0)
    #kappa['e'] = kappa_e
    #kappa['i'] = kappa_i
    # TODO

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
    if MMS_FLAG: a, L = add_mms_terms(a, L, v, dx, dS, ds, I_ch,
            phi_M_prev, mms, physical_params, mem_models, ion_list)

    return a, L
