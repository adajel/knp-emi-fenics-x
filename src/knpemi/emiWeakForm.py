import numpy as np
import dolfinx
import scifem
from mpi4py import MPI
from petsc4py import PETSc

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
    SpatialCoordinate,
    sin,
    pi,
    cos,
    div,
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
    phi_M_prev = {}

    for tag, subdomain in subdomain_list.items():
        mesh_sub = subdomain["mesh_sub"]
        # Create local functionspace for local potential ..
        V = dolfinx.fem.functionspace(mesh_sub, ("CG", degree))
        # ... and create and name function for local potential.
        phi[tag] = dolfinx.fem.Function(V)
        phi[tag].name = f"phi_{tag}"

        # Create membrane potential for all cellular subdomains (i.e. all subdomain but ECS)
        if tag > 0:
            mesh_mem = subdomain["mesh_mem"]

            # Create function space over gamma for membrane potential (phi_M)
            Q = dolfinx.fem.functionspace(mesh_mem, ("CG", degree))
            # Previous membrane potential
            phi_M_prev[tag] = dolfinx.fem.Function(Q)
            phi_M_prev[tag].name = f"phi_M_{tag}"

    return phi, phi_M_prev


def initialize_variables(ion_list, subdomain_list, c_prev, physical_params):
    """ Calculate kappa (tissue conductance) and set Nernst potentials """
    # Get physical parameters
    F = physical_params['F']
    psi = physical_params['psi']

    # Initialize dictionary
    kappa = {}
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        # Initialize kappa for each subdomain
        kappa_sub = 0
        # For each ion ...
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c_tag = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add contribution to kappa (tissue conductance)
            kappa_sub += F * ion['z'] * ion['z'] * ion['D'][tag] * psi * c_tag

            # Calculate and set Nernst potential for all cellular subdomains (i.e. all subdomain but ECS)
            if tag > 0:
                # ECS concentration (ECS is subdomain with tag 0)
                c_e = ion_list[-1][f'c_0'] if is_last else c_prev[0][idx]
                # Calculate and set Nernst potential
                ion[f'E_{tag}'] = 1 / (psi * ion['z']) * ln(c_e(e_res) / c_tag(i_res))

        # Add to dictionary
        kappa[tag] = kappa_sub

    # Calculate sum of ion specific channel currents for each cell (all
    # subdomain but the ECS) and each membrane model
    I_ch = {}
    for tag, subdomain in subdomain_list.items():
        if tag > 0:
            #tag = subdomain['tag']

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


def create_lhs(us, vs, dx, dS, subdomain_list, physical_params, kappa, splitting_scheme):
    """ Setup left hand side of the variational form for the emi system """
    C_phi = physical_params['C_phi']
    a = 0
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        # Get test and trial functions
        u = us[tag]; v = vs[tag]

        # Add contribution of subdomain to equation for potential (drift terms)
        a += inner(kappa[tag] * grad(u), grad(v)) * dx(tag)

        # Add membrane dynamics for each cellular subdomain (i.e. all subdomain but ECS)
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
                   - C_phi * (u_i(i_res) - u_e(e_res)) * v_e(e_res) * dS[tag_mm]

    return a

def create_prec(us, vs, dx, subdomain_list, kappa):
    """ Get preconditioner """
    p = 0
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        # Get test and trial functions
        u = us[tag]; v = vs[tag]
        p += kappa[tag] *  inner(grad(u), grad(v)) * dx(tag)
        # Add mass matrix for each cellular subdomain (i.e. all subdomain but ECS)
        if tag > 0:
            p += inner(u, v) * dx(tag)

    return p


def create_rhs(c_prev, vs, dx, dS, ion_list, subdomain_list, physical_params, 
        phi_M_prev, I_ch, splitting_scheme):
    """ Setup right hand side of variational form for the EMI system """

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    L = 0
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        v = vs[tag]
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c_tag = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'][tag])*grad(c_tag), grad(v)) * dx(tag)

        # Add membrane dynamics for each cellular subdomain (i.e. all subdomain but ECS)
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


def create_rhs_mms(vs, dx, dS, ds, c_prev, ion_list, subdomain_list,
        physical_params, dt, n, mms):

    C_phi = physical_params['C_phi']
    F = physical_params['F']
    L = 0
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        v = vs[tag]
        for idx, ion in enumerate(ion_list):
            # Determine the function source based on the index
            is_last = (idx == len(ion_list) - 1)
            c_tag = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'][tag])*grad(c_tag), grad(v)) * dx(tag)

            # Add Neumann term to ECS subdomain (zero in physiological simulation)
            if tag == 0: L += - F * ion['z'] * dot(ion['J_k_e'], n) * v * ds(5)

        # EMI source terms for potentials
        if tag == 0: L += inner(mms['f_phi_e'], v) * dx(tag)
        if tag == 1: L += inner(mms['f_phi_i'], v) * dx(tag)

        # Add membrane dynamics for each cellular subdomain (i.e. all subdomain but ECS)
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

def emi_system(mesh, ct, ft, physical_params, ion_list, subdomain_list,
        phi, phi_M_prev, c_prev, dt, degree=1, splitting_scheme=True, mms=None):
    """ Create and return EMI weak formulation """

    # Set MMS flag
    MMS_FLAG = False if mms is None else True
    # If MMS (i.e. no ODEs to solve), set splitting_scheme to false
    if MMS_FLAG: splitting_scheme = False

    # Create function-space for each subdomain
    V_list = []
    for tag, subdomain in subdomain_list.items():
        V_list.append(phi[tag].function_space)

    # Create mixed function space for potentials (phi)
    W = MixedFunctionSpace(*V_list)
    # Create trial and test functions
    us_ = TrialFunctions(W)
    vs_ = TestFunctions(W)

    # Add test and trial function to dictionary with subdomain tags as keys
    us = {}; vs = {}
    for idx, (tag, subdomain) in enumerate(subdomain_list.items()):
        us[tag] = us_[idx]
        vs[tag] = vs_[idx]

    # Create measures and facet normal
    dx, dS, ds = create_measures(mesh, ct, ft)
    n = FacetNormal(mesh)

    # Create tissue conductance and set Nernst potentials
    kappa, I_ch = initialize_variables(
            ion_list, subdomain_list, c_prev, physical_params
    )

    # Create standard variational formulation
    a = create_lhs(
            us, vs, dx, dS, subdomain_list, physical_params, kappa, splitting_scheme
    )

    # Create preconditioner
    p = create_prec(
            us, vs, dx, subdomain_list, kappa
    )

    L = create_rhs(
            c_prev, vs, dx, dS, ion_list, subdomain_list, physical_params,
            phi_M_prev, I_ch, splitting_scheme
    )

    # add terms specific to mms test
    if MMS_FLAG:
        L = create_rhs_mms(
                vs, dx, dS, ds, c_prev, ion_list, subdomain_list, 
                physical_params, dt, n, mms
        )

        # Create Dirichlet BC
        omega_e = subdomain_list[0]['mesh_sub']
        e_vertex_to_parent = subdomain_list[0]['sub_vertex_to_parent']
        exterior_to_parent = subdomain_list[0]['sub_to_parent']
        boundary_marker = 5
        sub_tag, _ = scifem.transfer_meshtags_to_submesh(
            ft, omega_e, e_vertex_to_parent, exterior_to_parent
        )

        omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
        bc_dofs = dolfinx.fem.locate_dofs_topological(
            Vs[0], omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
        )

        u_bc = dolfinx.fem.Function(Vs[0])
        u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
        bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)

        return a, p, L, dx, bc

    return a, p, L

    """

    omega = mesh

    omega_e = subdomain_list[0]['mesh_sub']
    omega_i = subdomain_list[1]['mesh_sub']

    interior_marker = subdomain_list[1]['tag']
    exterior_marker = subdomain_list[0]['tag']

    e_vertex_to_parent = subdomain_list[0]['sub_vertex_to_parent']
    exterior_to_parent = subdomain_list[0]['sub_to_parent']

    i_vertex_to_parent = subdomain_list[1]['sub_vertex_to_parent']
    interior_to_parent = subdomain_list[1]['sub_to_parent']

    interface_marker = 1
    boundary_marker = 5

    Ve = Vs[0]
    Vi = Vs[1]

    ve, vi = v[0], v[1]
    ue, ui = u[0], u[1]

    dGamma = dS[1]

    tr_ui = ui(i_res)
    tr_ue = ue(e_res)
    tr_vi = vi(i_res)
    tr_ve = ve(e_res)

    #x, y = SpatialCoordinate(omega)

    sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
    sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
    #Expression
    #sigma_i_exp = dolfinx.fem.Expression(sin(x), Vi.element.interpolation_points)
    #sigma_e_exp = dolfinx.fem.Expression(sin(x), Ve.element.interpolation_points)

    #sigma_i = dolfinx.fem.Function(Vi)
    #sigma_i.interpolate(sigma_i_exp)
    #sigma_e = dolfinx.fem.Function(Ve)
    #sigma_e.interpolate(sigma_e_exp)

    kappa = {0:sigma_e, 1:sigma_i}

    Cm = dolfinx.fem.Constant(omega, 1.0)
    dt = dolfinx.fem.Constant(omega, 1.0)

    ui_exact = mms['phi_i_exact']
    ue_exact = mms['phi_e_exact']

    n_e = n(e_res)
    n_i = n(i_res)

    Im_e = kappa[0] * inner(grad(ue_exact), n_e)
    Im_i = kappa[1] * inner(grad(ui_exact), n_i)
    g = Im_e + Im_i

    T = Cm / dt
    f = ui_exact - ue_exact - 1 / T * Im_e
    f_e = -div(kappa[0] * grad(ue_exact))
    f_i = -div(kappa[1] * grad(ui_exact))

    # get standard variational formulation
    a = create_lhs(
            u, v, dx, dS, subdomain_list, physical_params, kappa, splitting_scheme
    )

    L = T * inner(f, (tr_vi - tr_ve)) * dGamma
    L += f_e * ve * dx(0)
    L += f_i * vi * dx(1)
    L += inner(g, vi(i_res)) * dGamma

    sub_tag, _ = scifem.transfer_meshtags_to_submesh(
        ft, omega_e, e_vertex_to_parent, exterior_to_parent
    )
    omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
    bc_dofs = dolfinx.fem.locate_dofs_topological(
        Ve, omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
    )
    u_bc = dolfinx.fem.Function(Ve)
    u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)

    #ui = dolfinx.fem.Function(Vi, name="ui")
    #ue = dolfinx.fem.Function(Ve, name="ue")
    #entity_maps = [interior_to_parent, exterior_to_parent]

    #petsc_options = {
    #                "ksp_type": "preonly",
    #                "pc_type": "lu",
    #                "pc_factor_mat_solver_type": "mumps",
    #                "ksp_monitor": None,
    #                "ksp_error_if_not_converged": True,
    #            }

    #problem = dolfinx.fem.petsc.LinearProblem(
    #    extract_blocks(a),
    #    extract_blocks(L),
    #    u=[ue, ui],
    #    bcs=[bc],
    #    petsc_options=petsc_options,
    #    petsc_options_prefix="primal_single_",
    #    entity_maps=entity_maps,
    #)
    ##A = problem.A
    ##nullspace = PETSc.NullSpace().create(constant=True, comm=omega.comm)
    ##A.setNullSpace(nullspace)

    #problem.solve()

    #num_iterations = problem.solver.getIterationNumber()
    #converged_reason = problem.solver.getConvergedReason()
    #PETSc.Sys.Print(f"Solver converged in: {num_iterations} with reason {converged_reason}")

    #error_ui = dolfinx.fem.form(
    #    inner(ui - ui_exact, ui - ui_exact) * dx(1), entity_maps=entity_maps
    #)
    #error_ue = dolfinx.fem.form(
    #    inner(ue - ue_exact, ue - ue_exact) * dx(0), entity_maps=entity_maps
    #)
    #L2_ui = np.sqrt(scifem.assemble_scalar(error_ui, entity_maps=entity_maps))
    #L2_ue = np.sqrt(scifem.assemble_scalar(error_ue, entity_maps=entity_maps))
    #PETSc.Sys.Print(f"L2(ui): {L2_ui:.2e}\nL2(ue): {L2_ue:.2e}")

    return a, L, dx, bc
    """
