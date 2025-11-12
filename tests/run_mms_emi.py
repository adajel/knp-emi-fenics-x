from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp
from knpemi.utils import set_initial_conditions, setup_membrane_model

from knpemi.pdeSolver import create_solver_emi

from knpemi.utils import interpolate_to_membrane

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
    subdomain_tags = [0, 1]

    mesh_sub_1, i_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ct, interior_marker
    )

    mesh_sub_0, e_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ct, exterior_marker
    )

    mesh_g, g_to_parent, g_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ft, interface_marker
    )

    meshes = {"mesh":mesh, "mesh_sub_0":mesh_sub_0, "mesh_sub_1":mesh_sub_1, "mesh_g":mesh_g,
              "ct":ct, "ft":ft, "e_to_parent":e_to_parent,
              "i_to_parent":i_to_parent, "g_to_parent":g_to_parent,
              "e_vertex_to_parent": e_vertex_to_parent,
              "subdomain_tags":subdomain_tags}

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

    z_a = 1.0
    z_b = -1.0
    z_c = 1.0

    # Facet normal (pointing out of domain Omega) thus at gamma we have that
    # n_i = n and that n_e = -n.
    n = FacetNormal(mesh)
    n_i = n(i_res)

    # set background charge (no background charge in this scenario)
    rho = {0:dolfinx.fem.Constant(mesh_sub_0, 0.0),
           1:dolfinx.fem.Constant(mesh_sub_1, 0.0)}

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
    a_i_exact = 2 * x * y #sin(2 * pi * y) * cos(2 * pi * x)
    b_i_exact = 10 * y * y #cos(2 * pi * y) * sin(2 * pi * x)

    phi_i_exact = sin(2 * pi * x) * cos(2 * pi * y)
    phi_e_exact = sin(2 * pi * x) * cos(2 * pi * y)

    a_e_exact = 8 * x * x
    b_e_exact = 4 * y * y * y

    c_i_exact = - 1/z_c * (z_a * a_i_exact + z_b * b_i_exact)
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

    # Equation for the membrane potential source term (what we call robin):
    # g(x) = C phi_M - I_M (we here assume no membrane
    # currents i.e. Ich = 0 for MMS case) and we choose Im = F * sum_k(z^k *
    # dot(J_i_k, n_i)).
    f_phi_m = phi_M_exact - 1/C_phi * Im_intra

    # Source term continuity coupling condition on gamma i.e.
    # coupling condition for Im: Im_intra = - Im_extra + f which yields
    # f = Im_intra + Im_extra
    f_I_M = Im_intra + Im_extra

    # diffusion coefficients for each sub-domain
    D_a = {0:dolfinx.fem.Constant(mesh_sub_0, D_a_e),
           1:dolfinx.fem.Constant(mesh_sub_1, D_a_i)}

    D_b = {0:dolfinx.fem.Constant(mesh_sub_0, D_b_e),
           1:dolfinx.fem.Constant(mesh_sub_1, D_b_i)}

    D_c = {0:dolfinx.fem.Constant(mesh_sub_0, D_c_e),
           1:dolfinx.fem.Constant(mesh_sub_1, D_c_i)}

    # Create ions (channel conductivity is set below for each model)
    a = {'z':z_a,
         'name':'a',
         'D':D_a,
         'J_k_e':J_a_e}

    b = {'z':z_b,
         'name':'b',
         'D':D_b,
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

    V_e = phi[0].function_space
    V_i = phi[1].function_space

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
    mem_models = [{'ode':mm_mms_1, 'I_ch_k':{'a':0.0, 'b':0.0, 'c':0.0}}]

    # Create new cell marker on gamma mesh
    cell_marker = 1
    cell_map_g = mesh_g.topology.index_map(mesh_g.topology.dim)
    num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts

    # Create variational form emi problem
    a_emi, L_emi, dx, bc = emi_system(
            meshes, ct, ft, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev, c_prev, dt, mms=mms,
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [g_to_parent, e_to_parent, i_to_parent]

    # Create solver emi problem
    #problem_emi = create_solver_emi(a_emi, L_emi, phi, entity_maps, comm, bcs=[bc])
    problem_emi = create_solver_emi(a_emi, L_emi, phi, entity_maps, comm)
    problem_emi.solve()

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

if __name__ == "__main__":
    for n in [2, 3, 4, 5]:
        solve_system(n)
