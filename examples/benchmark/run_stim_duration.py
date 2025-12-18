#!/usr/bin/python3

from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp

from knpemi.pdeSolver import create_solver_emi
from knpemi.pdeSolver import create_solver_knp

from knpemi.utils import set_initial_conditions, setup_membrane_model
from knpemi.utils import interpolate_to_membrane

import mm_glial as mm_glial

import dolfinx
import adios4dolfinx
import scifem
from mpi4py import MPI
import numpy as np

from ufl import (
        ln,
        SpatialCoordinate,
        conditional,
        And,
        lt,
        gt,
)

# Define colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

i_res = "-"
e_res = "+"

comm = MPI.COMM_WORLD


def update_ode_variables(ode_model, c_prev, phi_M_prev, ion_list,
        subdomain_list, mesh, ct, tag, k):
    """ Update parameters in ODE solver (based on previous PDEs step) """
    # Get function space on membrane (gamma interface)
    Q = phi_M_prev.function_space

    # Set traces of ECS and ICS concentrations on membrane (from PDE solver) in ODE solver
    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_e = ion_list[-1]['c_0'] if is_last else c_prev[0][idx]
        c_i = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

        # Get and set extra and intracellular traces
        k_e, k_i = interpolate_to_membrane(c_e, c_i, Q, mesh, ct, subdomain_list, tag)
        ode_model.set_parameter(f"{ion['name']}_e", k_e)
        ode_model.set_parameter(f"{ion['name']}_i", k_i)

    # If first time step do nothing (the initial value for phi_M in ODEs are
    # taken from ODE file). For all other time steps, update the membrane
    # potential in ODE solver based on previous PDEs step
    if k > 0: ode_model.set_membrane_potential(phi_M_prev)

    return


def update_pde_variables(c, c_prev, phi, phi_M_prev, physical_parameters,
        ion_list, subdomain_list, mesh, ct):
    """ Update parameters in PDE solver for next time step """
    # Number of ions to solve for
    N_ions = len(ion_list[:-1])
    # Get physical parameters
    psi = physical_parameters['psi']
    rho = physical_parameters['rho']

    for tag, subdomain in subdomain_list.items():
        # Add contribution from immobile ions to eliminated ion
        c_elim_sum = - (1.0 / ion_list[-1]['z']) * rho['z'] * rho[tag]

        for idx, ion in enumerate(ion_list[:-1]):
            # Update previous concentration and scatter forward
            c_prev[tag][idx].x.array[:] = c[tag][idx].x.array
            c_prev[tag][idx].x.scatter_forward()

            # Add contribution to eliminated ion from ion concentration
            c_prev_sub = c_prev[tag][idx]
            c_elim_sum += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_prev_sub

        # Interpolate ufl sum onto V
        V = c[tag][idx].function_space
        c_elim = dolfinx.fem.Function(V)
        expr = dolfinx.fem.Expression(c_elim_sum, V.element.interpolation_points)
        c_elim.interpolate(expr)

        # Update eliminated ion concentration
        ion_list[-1][f'c_{tag}'].x.array[:] = c_elim.x.array

        # Update Nernst potentials for all cells (i.e. all subdomain but ECS)
        if tag != 0:
            # Update Nernst potentials for each ion we solve for
            for idx, ion in enumerate(ion_list[:-1]):
                # Get previous extra and intracellular concentrations
                c_e = c_prev[0][idx]
                c_i = c_prev[tag][idx]
                # Update Nernst potential
                ion['E'] =  1 / (psi * ion['z']) * ln(c_e(e_res) / c_i(i_res))

            # Update Nernst potential for eliminated ion
            c_e_elim = ion_list[-1][f'c_0']
            c_i_elim = ion_list[-1][f'c_{tag}']
            ion_list[-1]['E'] = 1 / (psi * ion['z']) * ln(c_e_elim(e_res) / c_i_elim(i_res))

            # Update membrane potential
            phi_e = phi[0]
            phi_i = phi[tag]
            phi_M_prev_sub = phi_M_prev[tag]

            # Update previous membrane potential (source term PDEs)
            Q = phi_M_prev_sub.function_space
            tr_phi_e, tr_phi_i = interpolate_to_membrane(phi_e, phi_i, Q, mesh, ct, subdomain_list, tag)
            phi_M_prev[tag].x.array[:] = tr_phi_i.x.array - tr_phi_e.x.array
            phi_M_prev[tag].x.scatter_forward()

    return


def write_to_file_sub(xdmf, fname, tag, phi, c, ion_list, t):
    # Write potentials to file
    xdmf.write_function(phi[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi[tag], time=float(t))

    # Write bulk concentrations to file
    for idx in range(len(ion_list)):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_tag = ion_list[-1][f'c_{tag}'] if is_last else c[tag][idx]

        # Write concentration to file
        xdmf.write_function(c_tag, t=float(t))
        adios4dolfinx.write_function(fname, c_tag, time=float(t))

    return


def write_to_file_mem(xdmf, fname, tag, mesh, ct, ion_list, subdomain_list, phi_M, c, t):
    # Write membrane potential to file
    xdmf.write_function(phi_M[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi_M[tag], time=float(t))

    # Write traces of concentrations on membrane to file
    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_e = ion_list[-1]['c_0'] if is_last else c[0][idx]
        c_i = ion_list[-1][f'c_{tag}'] if is_last else c[tag][idx]
        # Get extra and intracellular traces
        Q = phi_M[tag].function_space
        k_e, k_i = interpolate_to_membrane(c_e, c_i, Q, mesh, ct, subdomain_list, tag)
        # Write to file
        xdmf.write_function(k_e, t=float(t))
        xdmf.write_function(k_i, t=float(t))
        adios4dolfinx.write_function(fname, k_e, time=float(t))
        adios4dolfinx.write_function(fname, k_i, time=float(t))

    return


def solve_odes(c_prev, phi_M_prev, ion_list, stim_params, dt,
        mesh, ct, subdomain_list, k):
    """ Solve ODEs (membrane models) for each membrane tag in each subdomain """

    # Solve ODEs for all cells (i.e. all subdomains but the ECS)
    for tag, subdomain in subdomain_list.items():
        if tag > 0:
            # Get tag and membrane potential
            phi_M_prev_sub = phi_M_prev[tag]
            for mem_model in subdomain['mem_models']:
                # Update ODE variables based on PDE output
                ode_model = mem_model['ode']
                update_ode_variables(
                    ode_model, c_prev, phi_M_prev_sub, ion_list, subdomain_list,
                    mesh, ct, tag, k
                )

                # Solve ODEs
                ode_model.step_lsoda(
                        dt=dt,
                        stimulus=stim_params['stimulus'],
                        stimulus_locator=stim_params['stimulus_locator']
                )

                # Update PDE variables based on ODE output
                ode_model.get_membrane_potential(phi_M_prev_sub)

                # Update src terms for next PDE step based on ODE output
                for ion, I_ch_k in mem_model['I_ch_k'].items():
                    ode_model.get_parameter("I_ch_" + ion, I_ch_k)

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


def solve_system():
    """ Solve system (PDEs and ODEs) """
    # Read mesh and create sub-meshes for extra and intracellular domains and
    # for cellular membranes / interfaces (for solving ODEs)
    mesh_path = 'meshes/remarked_mesh/mesh.xdmf'

    mesh, ct, ft = read_mesh(mesh_path)

    # Subdomain tags (same as is mesh). NB! ECS tag must always be zero.
    ECS_tag = 0
    glial_tag = 1

    # Extract sub-meshes
    mesh_sub_0, sub_to_parent_0, sub_vertex_to_parent_0, _, _ = scifem.extract_submesh(mesh, ct, ECS_tag)
    mesh_sub_1, sub_to_parent_1, sub_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ct, glial_tag)
    mesh_mem_1, mem_to_parent_1, mem_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ft, glial_tag)

    # Create subdomains (extracellular space and cells)
    ECS = {"name":"ECS",
           "mesh_sub":mesh_sub_0,
           "sub_to_parent":sub_to_parent_0,
           "sub_vertex_to_parent":sub_vertex_to_parent_0}

    glial = {"name":"glial",
             "mesh_sub":mesh_sub_1,
             "sub_to_parent":sub_to_parent_1,
             "sub_vertex_to_parent":sub_vertex_to_parent_1,
             "mesh_mem":mesh_mem_1,
             "mem_to_parent":mem_to_parent_1}

    subdomain_list = {ECS_tag:ECS, glial_tag:glial}

    # Time variables (PDEs)
    t = dolfinx.fem.Constant(mesh, 0.0)
    dt = 0.1                         # global time step (ms)
    Tstop = 2                        # ms
    n_steps_ODE = 25                 # number of ODE steps

    # Spatial coordinates
    x, y, z = SpatialCoordinate(mesh)

    # Physical parameters
    C_M = 1.0                        # capacitance
    temperature = 307e3              # temperature (mK)
    F = 96500e3                      # Faraday's constant (mC/mol)
    R = 8.315e3                      # Gas Constant (mJ/(K*mol))
    D_Na = 1.33e-8                   # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-8                    # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-8                   # diffusion coefficients Cl (cm/ms)
    psi = F / (R * temperature)      # shorthand
    C_phi = C_M / dt                 # shorthand

    # Initial values
    K_e_init = 3.092970607490389
    K_n_init = 124.13988964240784
    K_g_init = 99.3100014897692
    Na_e_init = 144.60625137617149
    Na_n_init = 12.850454639128186
    Na_g_init = 15.775818906083778
    Cl_e_init = 133.62525154406637
    Cl_n_init = 5.0
    Cl_g_init = 5.203660274163705

    Cl_e_init = Na_e_init + K_e_init
    Cl_n_init = Na_n_init + K_n_init
    Cl_g_init = Na_g_init + K_g_init

    lambda_e = 0.5
    lambda_i = 3.4

    # Set background charge / immobile ions
    rho_z = -1
    rho_e = rho_z * (Na_e_init + K_e_init - Cl_e_init)
    rho_g = rho_z * (Na_g_init + K_g_init - Cl_g_init)
    rho_n = rho_z * (Na_n_init + K_n_init - Cl_n_init)

    rho = {'z':rho_z,
           0:dolfinx.fem.Constant(mesh_sub_0, rho_e),
           1:dolfinx.fem.Constant(mesh_sub_1, rho_g)}

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

    # diffusion coefficients for each sub-domain
    D_Na_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_Na/(lambda_e**2)),
                1:dolfinx.fem.Constant(mesh_sub_1, D_Na/(lambda_i**2))}
    D_K_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_K/(lambda_e**2)),
               1:dolfinx.fem.Constant(mesh_sub_1, D_K/(lambda_i**2))}
    D_Cl_sub = {0:dolfinx.fem.Constant(mesh_sub_0, D_Cl/(lambda_e**2)),
                1:dolfinx.fem.Constant(mesh_sub_1, D_Cl/(lambda_i**2))}

    # initial concentrations for each sub-domain
    Na_init = {0:dolfinx.fem.Constant(mesh_sub_0, Na_e_init), \
               1:dolfinx.fem.Constant(mesh_sub_1, Na_g_init)}
    K_init = {0:dolfinx.fem.Constant(mesh_sub_0, K_e_init), \
              1:dolfinx.fem.Constant(mesh_sub_1, K_g_init)}
    Cl_init = {0:dolfinx.fem.Constant(mesh_sub_0, Cl_e_init), \
               1:dolfinx.fem.Constant(mesh_sub_1, Cl_g_init)}

    # Region in which to apply the source term
    x_L = 2100e-7; x_U = 2900e-7
    y_L = 2100e-7; y_U = 2900e-7
    z_L = 2100e-7; z_U = 2500e-7

    # Strength of stimuli
    f_value = 500

    # Define when (t) and where (x, y, z) source term is applied
    f_condition = And(gt(t, 0.2),
                  And(lt(t, 1.2),
                  And(gt(x, x_L),
                  And(lt(x, x_U),
                  And(lt(y, y_U),
                  And(gt(y, y_L),
                  And(gt(z, z_L), lt(z, z_U))))))))

    # Define source terms
    f_source_K = conditional(f_condition, f_value, 0)
    f_source_Na = conditional(f_condition, - f_value, 0)

    # Create ions (channel conductivity is set below in the membrane model)
    Na = {'c_init':Na_init,
          'bdry':dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z': 1.0,
          'name':'Na',
          'D':D_Na_sub,
          'f_source':f_source_Na}

    K = {'c_init':K_init,
          'bdry':dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
         'z':1.0,
         'name':'K',
         'D':D_K_sub,
         'f_source':f_source_K}

    Cl = {'c_init':Cl_init,
          'bdry':dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z': - 1.0,
          'name':'Cl',
          'D':D_Cl_sub}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    phi, phi_M_prev = create_functions_emi(subdomain_list, degree=1)
    c, c_prev = create_functions_knp(subdomain_list, ion_list, degree=1)

    # Set initial conditions for PDE system
    set_initial_conditions(ion_list, subdomain_list, c_prev)

    # Create new cell marker on gamma mesh
    #cell_marker = 1
    #cell_map_g = mesh_mem_1.topology.index_map(mesh_mem_1.topology.dim)
    #num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts

    # Transfer mesh tags from ct to tags for gamma mesh on interface
    #ct_g = dolfinx.mesh.meshtags(
        #mesh_mem_1, mesh_mem_1.topology.dim, np.arange(num_cells_local, dtype=np.int31), cell_marker
    #)

    ct_g_1, _ = scifem.transfer_meshtags_to_submesh(
            ft, mesh_mem_1, mem_vertex_to_parent_1, mem_to_parent_1
    )

    # Dictionary with mesh tags (key is facet tag)
    ct_g = {1: ct_g_1}

    # Dictionary with membrane function spaces (key is facet tag)
    Q = {1: phi_M_prev[glial_tag].function_space}

    # Synaptic conductivity (S/m**2)
    g_syn_bar = 0.0
    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: True

    # Set membrane parameters
    stim_params = {'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    # TODO: make clearer, better design?
    # Dictionary with membrane / ode models (key is facet tag) NB! New tags,
    # now for membranes (not cells)!!! Should we change to 3, 4, i.e. 1->3 and
    # 2->4 to be clear on diff?
    ode_models_glial = {1: mm_glial}
    mem_models_glial = setup_membrane_model(
            stim_params, physical_parameters, ode_models_glial,
            ct_g[glial_tag], Q[glial_tag], ion_list
    )

    # Add membrane model to neuron in subdomain list
    subdomain_list[1]['mem_models'] = mem_models_glial

    # Create variational form emi problem
    a_emi, p_emi, L_emi = emi_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            phi, phi_M_prev, c_prev, dt,
    )

    # Create variational form knp problem
    a_knp, p_knp, L_knp = knp_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            phi, phi_M_prev, c, c_prev, dt,
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [mem_to_parent_1, sub_to_parent_0, sub_to_parent_1]

    # Create direct solver emi problem
    problem_emi = create_solver_emi(
            a_emi, L_emi, phi, entity_maps, subdomain_list, comm
    )

    """
    # Create iterative solver emi problem
    problem_emi = create_solver_emi(
            a_emi, L_emi, phi, entity_maps, subdomain_list, comm, direct=False,
            p=p_emi
    )
    """

    # Create solver knp problem
    problem_knp = create_solver_knp(
            a_knp, L_knp, c, entity_maps, subdomain_list, comm
    )

    # Crate dictionary for storing XDMF files and checkpoint filenames
    xdmf_sub = {}; xdmf_mem = {}
    fname_bp_sub = {}; fname_bp_mem = {}

    fname = "benchmark"

    # Create files (XDMF and checkpoint) for saving results
    for tag, subdomain in subdomain_list.items():
        xdmf = dolfinx.io.XDMFFile(comm, f"results/{fname}/results_sub_{tag}.xdmf", "w")
        xdmf.write_mesh(subdomain['mesh_sub'])
        adios4dolfinx.write_mesh(f"results/{fname}/checkpoint_sub_{tag}.bp", subdomain['mesh_sub'])
        xdmf_sub[tag] = xdmf
        fname_bp_sub[tag] = f"results/{fname}/checkpoint_sub_{tag}.bp"

        # Write membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
        if tag > 0:
            xdmf = dolfinx.io.XDMFFile(comm, f"results/{fname}/results_mem_{tag}.xdmf", "w")
            xdmf.write_mesh(subdomain['mesh_mem'])
            adios4dolfinx.write_mesh(f"results/{fname}/checkpoint_mem_{tag}.bp", subdomain['mesh_mem'])
            xdmf_mem[tag] = xdmf
            fname_bp_mem[tag] = f"results/{fname}/checkpoint_mem_{tag}.bp"

    for k in range(int(round(Tstop/float(dt)))):
        print(f'solving for t={float(t)}')

        # Solve ODEs
        solve_odes(
                c_prev, phi_M_prev, ion_list, stim_params,
                dt, mesh, ct, subdomain_list, k
            )

        # Solve PDEs
        problem_emi.solve()
        problem_knp.solve()

        update_pde_variables(
                c, c_prev, phi, phi_M_prev, physical_parameters,
                ion_list, subdomain_list, mesh, ct,
        )

        # update time
        t.value = float(t + dt)

        # Write results to file
        for tag, subdomain in subdomain_list.items():
            # concentrations and potentials from previous time step to file
            write_to_file_sub(xdmf_sub[tag], fname_bp_sub[tag], tag, phi, c, ion_list, t)
            # membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
            if tag > 0:
                write_to_file_mem(xdmf_mem[tag], fname_bp_mem[tag], tag, mesh, ct, ion_list, subdomain_list, phi_M_prev, c, t)

    # Close XDMF files
    for tag, subdomain in subdomain_list.items():
        xdmf_sub[tag].close()
        if tag > 0:
            xdmf_mem[tag].close()

if __name__ == "__main__":
    solve_system()
