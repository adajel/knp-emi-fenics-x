from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp
from knpemi.utils import set_initial_conditions, setup_membrane_model

from knpemi.emiSolver import create_solver_emi
from knpemi.knpSolver import create_solver_knp

from knpemi.utils import interpolate_to_membrane

import mm_hh as mm_hh

import dolfinx
import scifem
from mpi4py import MPI
import numpy as np

from ufl import (
        ln,
        pi,
        sin,
        cos,
        div,
        grad,
        dot,
        diff,
        variable,
        SpatialCoordinate,
        FacetNormal,
        inner,
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

comm = MPI.COMM_WORLD

class MMSMembraneModel:
    pass

def update_pde_variables(c_prev, c, phi, phi_M_prev, physical_parameters, ion_list, meshes):
    # Number of ions to solve for
    N_ions = len(ion_list[:-1])

    # Get physical parameters
    temperature = physical_parameters['temperature']
    F = physical_parameters['F']
    R = physical_parameters['R']
    rho = physical_parameters['rho']

    phi_i = phi['i']
    phi_e = phi['e']

    # Update previous extra and intracellular concentrations
    for idx in range(N_ions):
        c_prev['e'][idx].x.array[:] = c['e'][idx].x.array
        c_prev['i'][idx].x.array[:] = c['i'][idx].x.array
        # Scatter
        c_prev['e'][idx].x.scatter_forward()
        c_prev['i'][idx].x.scatter_forward()

    # Update previous membrane potential (source term PDEs)
    Q = phi_M_prev.function_space
    tr_phi_e, tr_phi_i = interpolate_to_membrane(phi_e, phi_i, Q, meshes)
    phi_M_prev.x.array[:] = tr_phi_i.x.array - tr_phi_e.x.array
    phi_M_prev.x.scatter_forward()

    # Add contribution from background charge / immobile ions to eliminated ion
    c_e_elim_sum = - (1.0 / ion_list[-1]['z']) * rho[0]
    c_i_elim_sum = - (1.0 / ion_list[-1]['z']) * rho[1]

    # Update Nernst potentials for next global time level
    for idx, ion in enumerate(ion_list[:-1]):
        # Get previous extra and intracellular concentrations
        c_e = c_prev['e'][idx]
        c_i = c_prev['i'][idx]
        # Update Nernst potential
        ion['E'] = R * temperature / (F * ion['z']) * ln(c_e(e_res) / c_i(i_res))

        # Add ion specific contribution to eliminated ion concentration
        c_e_elim_sum += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_e
        c_i_elim_sum += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_i

    # Interpolate eliminated ion concentration sum onto function spaces
    V_e = c['e'][0].function_space
    V_i = c['i'][1].function_space

    c_e_elim = dolfinx.fem.Function(V_e)
    c_i_elim = dolfinx.fem.Function(V_i)

    expr_e = dolfinx.fem.Expression(c_e_elim_sum, V_e.element.interpolation_points)
    expr_i = dolfinx.fem.Expression(c_i_elim_sum, V_i.element.interpolation_points)

    c_e_elim.interpolate(expr_e)
    c_i_elim.interpolate(expr_i)

    # Update eliminated ion concentrations
    ion_list[-1]['c_e'].x.array[:] = c_e_elim.x.array
    ion_list[-1]['c_i'].x.array[:] = c_i_elim.x.array
    # Update Nernst potential for eliminated ion concentrations
    ion_list[-1]['E'] = R * temperature / (F * ion['z']) * ln(c_e_elim(e_res) / c_i_elim(i_res))

    return


def write_to_file(xdmf_e, xdmf_i, phi, c, phi_M, ion_list, t):

    # Write results to file
    #xdmf_e.write_function(phi['e'], t=float(t))
    #xdmf_i.write_function(phi['i'], t=float(t))
    #xdmf.write_function(phi_M, t=float(t))

    c_e = c['e'][0]
    xdmf_e.write_function(c_e, t=float(t))

    c_i = c['i'][0]
    xdmf_i.write_function(c_i, t=float(t))

    c_e = c['e'][1]
    xdmf_e.write_function(c_e, t=float(t))

    c_i = c['i'][1]
    xdmf_i.write_function(c_i, t=float(t))

    for idx in range(len(ion_list)):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)

        c_i = ion_list[-1]['c_i'] if is_last else c['i'][idx]
        c_e = ion_list[-1]['c_e'] if is_last else c['e'][idx]

        # Write the functions to file
        #(xdmf.write_function(f, t=float(t)) for f in (c_i, c_e))
        #xdmf_e.write_function(c_e, t=float(t))
        #xdmf_i.write_function(c_i, t=float(t))

    return

def read_mesh(mesh_file):

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    with dolfinx.io.XDMFFile(comm, mesh_file, 'r') as xdmf:
        # Read mesh and cell tags
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode)
        ct = xdmf.read_meshtags(mesh, name='cell_marker')

        # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
        mesh.topology.create_entities(mesh.topology.dim-1)
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

        # Read facets
        ft = xdmf.read_meshtags(mesh, name='facet_marker')

    xdmf.close()

    return mesh, ct, ft

def solve_system(resolution):
    """ Solve system (PDEs and ODEs) """
    # Read mesh and create sub-meshes for extra and intracellular domains and
    # for cellular membranes / interfaces (for solving ODEs)
    mesh_path = f'meshes/mms/mesh_{resolution}.xdmf'
    mesh, ct, ft = read_mesh(mesh_path)

    # subdomain markers
    exterior_marker = 0; interior_marker = 1,

    # gamma markers
    interface_marker = 1

    mesh_i, i_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ct, interior_marker
    )

    mesh_e, e_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ct, exterior_marker
    )

    mesh_g, g_to_parent, g_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ft, interface_marker
    )

    meshes = {"mesh":mesh, "mesh_e":mesh_e, "mesh_i":mesh_i, "mesh_g":mesh_g,
              "ct":ct, "ft":ft, "e_to_parent":e_to_parent,
              "i_to_parent":i_to_parent, "g_to_parent":g_to_parent,
              "e_vertex_to_parent": e_vertex_to_parent}

    # Time variables
    t = dolfinx.fem.Constant(mesh, 0.0) # time constant
    dt = 1.0                            # global time step (ms)
    Tstop = dt                          # global end time (ms)

    # Physical parameters
    C_M = 1.0
    temperature = 1.0
    F = 1.0
    R = 1.0
    D_a_i = 1.0
    D_b_i = 1.0
    D_c_i = 1.0
    D_a_e = 1.0
    D_b_e = 1.0
    D_c_e = 1.0
    psi = F / (R * temperature)
    C_phi = C_M / dt

    # Coupling coefficients
    C_a_i = 1.0
    C_b_i = 1.0
    C_c_i = 1.0
    C_a_e = 1.0
    C_b_e = 1.0
    C_c_e = 1.0

    z_a = 1.0
    z_b = -1.0
    z_c = 1.0

    # Facet normal (pointing out of domain Omega) thus at gamma we have that
    # n_i = n and that n_e = -n.
    n = FacetNormal(mesh)
    n_i = n(i_res)

    # set background charge (no background charge in this scenario)
    rho = {0:dolfinx.fem.Constant(mesh_e, 0.0),
           1:dolfinx.fem.Constant(mesh_i, 0.0)}

    # Set parameters
    physical_parameters = {'dt':dolfinx.fem.Constant(mesh, dt),
                           'n_steps_ODE':dolfinx.fem.Constant(mesh, dt),
                           'F':dolfinx.fem.Constant(mesh, F),
                           'psi':dolfinx.fem.Constant(mesh, psi),
                           'C_phi':dolfinx.fem.Constant(mesh, C_phi), 
                           'C_M':dolfinx.fem.Constant(mesh, C_M),
                           'R':dolfinx.fem.Constant(mesh, R),
                           'temperature':dolfinx.fem.Constant(mesh, temperature),
                           'rho':rho}

    x, y = SpatialCoordinate(mesh)

    # Define intracellular exact solutions
    a_i_exact = sin(2 * pi * y) * cos(2 * pi * x)
    b_i_exact = cos(2 * pi * y) * sin(2 * pi * x)
    c_i_exact = - 1/z_c * (z_a * a_i_exact + z_b * b_i_exact)

    phi_i_exact = cos(2 * pi * x) * cos(2 * pi * y)
    # Define extracellular exact solutions
    phi_e_exact = sin(2 * pi * x) * sin(2 * pi * y)

    a_e_exact = a_i_exact
    b_e_exact = b_i_exact
    c_e_exact = - 1/z_c * (z_a * a_i_exact + z_b * b_i_exact)

    # Exact membrane potential
    phi_M_exact = phi_i_exact - phi_e_exact

    # Shorthand for linearised ion fluxes
    J_a_i = - D_a_i * grad(a_i_exact) - z_a * D_a_i * psi * a_i_exact * grad(phi_i_exact)
    J_b_i = - D_b_i * grad(b_i_exact) - z_b * D_b_i * psi * b_i_exact * grad(phi_i_exact)
    J_c_i = - D_c_i * grad(c_i_exact) - z_c * D_c_i * psi * c_i_exact * grad(phi_i_exact)

    J_a_e = - D_a_e * grad(a_e_exact) - z_a * D_a_e * psi * a_e_exact * grad(phi_e_exact)
    J_b_e = - D_b_e * grad(b_e_exact) - z_b * D_b_e * psi * b_e_exact * grad(phi_e_exact)
    J_c_e = - D_c_e * grad(c_e_exact) - z_c * D_c_e * psi * c_e_exact * grad(phi_e_exact)

    # Potential source terms = F * sum_k(z_k * div(J_k_r))
    f_phi_i = F * (z_a * div(J_a_i) + z_b * div(J_b_i) + z_c * div(J_c_i))
    f_phi_e = F * (z_a * div(J_a_e) + z_b * div(J_b_e) + z_c * div(J_c_e))

    # Source terms per species f = dk_r/dt + div(J_k_r) with dk_r/dt = 0
    f_a_i = div(J_a_i)
    f_a_e = div(J_a_e)
    f_b_i = div(J_b_i)
    f_b_e = div(J_b_e)
    f_c_i = div(J_c_i)
    f_c_e = div(J_c_e)

    # Total intracellular membrane flux = F * sum_k(z^k * J_k_i)
    # and intracellular membrane currents I_M = Im_intra = dot(total_flux_intra, n_i)
    total_flux_intra = F * (z_a * J_a_i + z_b * J_b_i + z_c * J_c_i)
    Im_intra = dot(total_flux_intra, n_i)

    # Total extracellular membrane flux = F * sum_k(z^k * J_k_e)
    # and extracellular membrane currents
    # I_M = - Im_extra = - dot(total_flux_extra, n_e). We have that n = n_i
    # and that - n = n_e.
    total_flux_extra = F * (z_a * J_a_e + z_b * J_b_e + z_c * J_c_e)
    Im_extra = - dot(total_flux_extra, n_i)

    # Ion channel currents
    #Ich_a = phi_M_exact
    #Ich_b = phi_M_exact
    #Ich_c = phi_M_exact
    #Ich = Ich_a + Ich_b + Ich_c

    # ????????????????
    # Equation for the membrane potential source term (what we call robin):
    # g(x) = C phi_M - I_M (we here assume no membrane
    # currents i.e. Ich = 0 for MMS case) and we choose Im = F * sum_k(z^k *
    # dot(J_i_k, n_i)).
    f_phi_m_a_i = phi_M_exact - 1/C_a_i * dot(J_a_i, n_i)
    f_phi_m_b_i = phi_M_exact - 1/C_b_i * dot(J_b_i, n_i)
    f_phi_m_a_e = phi_M_exact - 1/C_a_e * dot(J_a_e, n_i)
    f_phi_m_b_e = phi_M_exact - 1/C_b_e * dot(J_b_e, n_i)

    f_phi_m = phi_M_exact - 1/C_phi * Im_intra

    # Source term continuity coupling condition on gamma i.e.
    # coupling condition for Im: Im_intra = - Im_extra + f which yields
    # f = Im_intra + Im_extra
    f_I_M = Im_intra + Im_extra

    # diffusion coefficients for each sub-domain
    D_a = {0:dolfinx.fem.Constant(mesh_e, D_a_e),
           1:dolfinx.fem.Constant(mesh_i, D_a_i)}

    D_b = {0:dolfinx.fem.Constant(mesh_e, D_b_e),
           1:dolfinx.fem.Constant(mesh_i, D_b_i)}

    D_c = {0:dolfinx.fem.Constant(mesh_e, D_c_e),
           1:dolfinx.fem.Constant(mesh_i, D_c_i)}

    # diffusion coefficients for each sub-domain
    C_a = {0:dolfinx.fem.Constant(mesh_e, C_a_e),
           1:dolfinx.fem.Constant(mesh_i, C_a_i)}

    C_b = {0:dolfinx.fem.Constant(mesh_e, C_b_e),
           1:dolfinx.fem.Constant(mesh_i, C_b_i)}

    C_c = {0:dolfinx.fem.Constant(mesh_e, C_c_e),
           1:dolfinx.fem.Constant(mesh_i, C_c_i)}

    # Create ions (channel conductivity is set below for each model)
    a = {'z':z_a,
         'name':'a',
         'D':D_a,
         'f_k_e':f_a_e,
         'f_k_i':f_a_i,
         'f_phi_m_e':f_phi_m_a_e,
         'f_phi_m_i':f_phi_m_a_i,
         'C':C_a,
         'f_source':f_a_e,
         'J_k_e':J_a_e}

    b = {'z':z_b,
         'name':'b',
         'D':D_b,
         'f_k_e':f_b_e,
         'f_k_i':f_b_i,
         'f_phi_m_e':f_phi_m_b_e,
         'f_phi_m_i':f_phi_m_b_i,
         'C':C_b,
         'f_source':f_b_e,
         'J_k_e':J_b_e}

    c = {'z':z_c,
         'name':'c',
         'D':D_c,
         'J_k_e':J_c_e}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [a, b, c]

    # Create dictionary for MM terms (source terms, boundary terms etc.)
    mms = {"f_phi_i" : f_phi_i,
           "f_phi_e" : f_phi_e,
           "f_phi_m" : f_phi_m,
           "f_I_M" : f_I_M,
           }

    # get functions
    phi, phi_M_prev = create_functions_emi(meshes, degree=1)
    c, c_prev = create_functions_knp(meshes, ion_list, degree=1)

    V_e = phi['e'].function_space
    V_i = phi['i'].function_space

    # Set initial conditions
    a_e_expr = dolfinx.fem.Expression(a_e_exact, V_e.element.interpolation_points)
    b_e_expr = dolfinx.fem.Expression(b_e_exact, V_e.element.interpolation_points)
    c_e_expr = dolfinx.fem.Expression(c_e_exact, V_e.element.interpolation_points)
    a_i_expr = dolfinx.fem.Expression(a_i_exact, V_i.element.interpolation_points)
    b_i_expr = dolfinx.fem.Expression(b_i_exact, V_i.element.interpolation_points)
    c_i_expr = dolfinx.fem.Expression(c_i_exact, V_i.element.interpolation_points)

    a_e_func = dolfinx.fem.Function(V_e)
    b_e_func = dolfinx.fem.Function(V_e)
    c_e_func = dolfinx.fem.Function(V_e)
    a_i_func = dolfinx.fem.Function(V_i)
    b_i_func = dolfinx.fem.Function(V_i)
    c_i_func = dolfinx.fem.Function(V_i)

    a_e_func.interpolate(a_e_expr)
    b_e_func.interpolate(b_e_expr)
    c_e_func.interpolate(c_e_expr)
    a_i_func.interpolate(a_i_expr)
    b_i_func.interpolate(b_i_expr)
    c_i_func.interpolate(c_i_expr)

    # Set initial concentrations for each sub-domain
    ion_list[0]['c_init'] = {0:a_e_func.x.array, 1:a_i_func.x.array}
    ion_list[1]['c_init'] = {0:b_e_func.x.array, 1:b_i_func.x.array}
    ion_list[2]['c_init'] = {0:c_e_func.x.array, 1:c_i_func.x.array}

    # Set initial conditions in solver
    set_initial_conditions(ion_list, c_prev)

    # Create dummy membrane models for MMS case
    mm_mms_1 = MMSMembraneModel(); mm_mms_1.tag = 1
    mem_models = [{'ode':mm_mms_1,
                  'I_ch_k':{'a':0.0, 'b':0.0, 'c':0.0}}]

    # Create new cell marker on gamma mesh
    cell_marker = 1
    cell_map_g = mesh_g.topology.index_map(mesh_g.topology.dim)
    num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts

    # Transfer mesh tags from ct to tags for gamma mesh on interface
    ct_g, _ = scifem.transfer_meshtags_to_submesh(
            ft, mesh_g, g_vertex_to_parent, g_to_parent
    )

    # Create variational form emi problem
    a_emi, L_emi, dx, bc = emi_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev, c_prev, dt, mms=mms,
    )

    """
    # ---------------------------------------
    # TODO
    V_e = phi['e'].function_space
    V_i = phi['i'].function_space
    phi['e'] = phi_e_exact
    phi['i'] = phi_i_exact
    phi_M_pre = phi_i_exact - phi_e_exact
    #f_phi_e_exact = dolfinx.fem.Function(V_e)
    #f_phi_i_exact = dolfinx.fem.Function(V_i)
    #expr_e = dolfinx.fem.Expression(phi_e_exact, V_e.element.interpolation_points)
    #expr_i = dolfinx.fem.Expression(phi_i_exact, V_i.element.interpolation_points)
    #f_phi_e_exact.interpolate(expr_e)
    #f_phi_i_exact.interpolate(expr_i)

    #Q = phi_M_prev.function_space
    #tr_phi_e, tr_phi_i = interpolate_to_membrane(f_phi_e_exact, f_phi_i_exact, Q, meshes)
    #phi_M_prev.x.array[:] = tr_phi_i.x.array - tr_phi_e.x.array
    # TODO
    # ---------------------------------------
    """

    # Create variational form knp problem
    a_knp, L_knp, dx = knp_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev, c, c_prev, dt, mms=mms,
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [g_to_parent, e_to_parent, i_to_parent]

    # Create solver emi problem
    problem_emi = create_solver_emi(a_emi, L_emi, phi, entity_maps, comm)
    #problem_emi = create_solver_emi(a_emi, L_emi, phi, entity_maps, comm, bcs=[bc])
    # Create solver knp problem
    problem_knp = create_solver_knp(a_knp, L_knp, c, entity_maps)

    xdmf_e = dolfinx.io.XDMFFile(mesh.comm, "results/results_e.xdmf", "w")
    xdmf_i = dolfinx.io.XDMFFile(mesh.comm, "results/results_i.xdmf", "w")

    # write mesh and mesh tags to file
    xdmf_e.write_mesh(mesh_e)
    xdmf_i.write_mesh(mesh_i)

    """
    for k in range(int(round(Tstop/float(dt)))):
        print(f'solving for t={float(t)}')

        # Solve PDEs
        #problem_emi.solve()
        problem_knp.solve()

        # update PDE variables
        update_pde_variables(
                c_prev, c, phi, phi_M_prev, physical_parameters,
                ion_list, meshes
        )

        # update time
        t.value = float(t + dt)

        # Write results from previous time step to file
        write_to_file(xdmf_e, xdmf_i, phi, c, phi_M_prev, ion_list, t)
    """
    problem_emi.solve()
    #problem_knp.solve()

    xdmf_e.close()
    xdmf_i.close()

    xdmf_e_exact = dolfinx.io.XDMFFile(mesh.comm, "results/results_e_exact.xdmf", "w")
    xdmf_e_exact.write_mesh(mesh_e)

    func_a_e_exaxt = dolfinx.fem.Function(V_e)
    expr_a_e = dolfinx.fem.Expression(a_e_exact, V_e.element.interpolation_points)
    func_a_e_exaxt.interpolate(expr_a_e)
    func_a_e_exaxt.name = "a_e_exact"

    xdmf_e_exact.write_function(func_a_e_exaxt, t=float(t))
    xdmf_e_exact.close()

    xdmf_a_e = dolfinx.io.XDMFFile(mesh.comm, "results/results_a_e.xdmf", "w")
    xdmf_a_e.write_mesh(mesh_e)
    xdmf_a_e.write_function(c['e'][0], t=float(t))
    xdmf_a_e.close()

    xdmf_b_e = dolfinx.io.XDMFFile(mesh.comm, "results/results_b_e.xdmf", "w")
    xdmf_b_e.write_mesh(mesh_e)
    xdmf_b_e.write_function(c['e'][1], t=float(t))
    xdmf_b_e.close()


    xdmf_i_exact = dolfinx.io.XDMFFile(mesh.comm, "results/results_i_exact.xdmf", "w")
    xdmf_i_exact.write_mesh(mesh_i)

    func_a_i_exaxt = dolfinx.fem.Function(V_i)
    expr_a_i = dolfinx.fem.Expression(a_i_exact, V_i.element.interpolation_points)
    func_a_i_exaxt.interpolate(expr_a_i)
    func_a_i_exaxt.name = "a_i_exact"

    xdmf_i_exact.write_function(func_a_i_exaxt, t=float(t))
    xdmf_i_exact.close()

    xdmf_a_i = dolfinx.io.XDMFFile(mesh.comm, "results/results_a_i.xdmf", "w")
    xdmf_a_i.write_mesh(mesh_i)
    xdmf_a_i.write_function(c['i'][0], t=float(t))
    xdmf_a_i.close()

    xdmf_b_i = dolfinx.io.XDMFFile(mesh.comm, "results/results_b_i.xdmf", "w")
    xdmf_b_i.write_mesh(mesh_i)
    xdmf_b_i.write_function(c['i'][1], t=float(t))
    xdmf_b_i.close()



    #phi = problem_emi.u
    #phi_e = phi['e']
    #phi_i = phi['i']
    phi_e, phi_i= problem_emi.u

    # calculate ICS error
    error_phi_i = dolfinx.fem.form(
    inner(phi_i - phi_i_exact, phi_i - phi_i_exact) * dx(1), entity_maps=entity_maps
    )

    # calculate ECS error
    error_phi_e = dolfinx.fem.form(
    inner(phi_e - phi_e_exact, phi_e - phi_e_exact) * dx(0), entity_maps=entity_maps
    )

    # get L2 norm of error
    L2_ui = np.sqrt(scifem.assemble_scalar(error_phi_i, entity_maps=entity_maps))
    L2_ue = np.sqrt(scifem.assemble_scalar(error_phi_e, entity_maps=entity_maps))

    print("phi_i", L2_ui)
    print("phi_e", L2_ue)

    c = problem_knp.u

    a_e = c[0]
    b_e = c[1]
    a_i = c[2]
    b_i = c[3]

    # calculate ICS error
    error_a_i = dolfinx.fem.form(
    inner(a_i - a_i_exact, a_i - a_i_exact) * dx(1), entity_maps=entity_maps
    )

    # calculate ECS error
    error_a_e = dolfinx.fem.form(
    inner(a_e - a_e_exact, a_e - a_e_exact) * dx(0), entity_maps=entity_maps
    )

    # get L2 norm of error
    L2_ui_a = np.sqrt(scifem.assemble_scalar(error_a_i, entity_maps=entity_maps))
    L2_ue_a = np.sqrt(scifem.assemble_scalar(error_a_e, entity_maps=entity_maps))

    print("a_i", L2_ui_a)
    print("a_e", L2_ue_a)

    # calculate ICS error
    error_b_i = dolfinx.fem.form(
    inner(b_i - b_i_exact, b_i - b_i_exact) * dx(1), entity_maps=entity_maps
    )

    # calculate ECS error
    error_b_e = dolfinx.fem.form(
    inner(b_e - b_e_exact, b_e - b_e_exact) * dx(0), entity_maps=entity_maps
    )

    # get L2 norm of error
    L2_ui_b = np.sqrt(scifem.assemble_scalar(error_b_i, entity_maps=entity_maps))
    L2_ue_b = np.sqrt(scifem.assemble_scalar(error_b_e, entity_maps=entity_maps))

    print("b_i", L2_ui_b)
    print("b_e", L2_ue_b)

if __name__ == "__main__":
    for n in [2, 3, 4]:
        solve_system(n)
