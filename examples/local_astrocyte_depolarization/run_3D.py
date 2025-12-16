from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp

from knpemi.pdeSolver import create_solver_emi
from knpemi.pdeSolver import create_solver_knp

from knpemi.utils import set_initial_conditions, setup_membrane_model
from knpemi.utils import interpolate_to_membrane

import mm_3D as mm_hh

import dolfinx
import adios4dolfinx
import scifem
from mpi4py import MPI
import numpy as np

from ufl import (
        ln,
)

from memory_profiler import profile

i_res = "-"
e_res = "+"

comm = MPI.COMM_WORLD

@profile
def solve_emi(problem_emi):
    problem_emi.solve()

@profile
def solve_knp(problem_knp):
    problem_knp.solve()

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

    for subdomain in subdomain_list:
        tag = subdomain['tag']
        # Add contribution from immobile ions to eliminated ion
        c_elim_sum = - (1.0 / ion_list[-1]['z']) * rho[tag]

        for idx, ion in enumerate(ion_list[:-1]):
            # Update previous concentration and scatter forward
            c_prev[tag][idx].x.array[:] = c[tag][idx].x.array
            c_prev[tag][idx].x.scatter_forward()

            # Add contribution to eliminated ion from ion concentration
            c_prev_sub = c_prev[tag][idx]
            c_elim_sum += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_prev_sub

        # Interpolate ufl sum onto V
        V = c[tag][tag].function_space
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
    # Write potential to file
    xdmf.write_function(phi[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi[tag], time=float(t))

    for idx in range(len(ion_list)):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_tag = ion_list[-1][f'c_{tag}'] if is_last else c[tag][idx]

        # Write concentration to file
        xdmf.write_function(c_tag, t=float(t))
        adios4dolfinx.write_function(fname, c_tag, time=float(t))

    return


def write_to_file_mem(xdmf, fname, tag, phi_M, t):
    # Write potential to file
    xdmf.write_function(phi_M[tag], t=float(t))
    adios4dolfinx.write_function(fname, phi_M[tag], time=float(t))

    return


def solve_odes(mem_models, c_prev, phi_M_prev, ion_list, stim_params, dt,
        mesh, ct, subdomain_list, k):
    """ Solve ODEs (membrane models) for each membrane tag in each subdomain """

    # Solve ODEs for all cells (i.e. all subdomains but the ECS)
    for subdomain in subdomain_list[1:]:
        # Get tag and membrane potential
        tag = subdomain['tag']
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


@profile
def solve_system():
    """ Solve system (PDEs and ODEs) """
    # Read mesh and create sub-meshes for extra and intracellular domains and
    # for cellular membranes / interfaces (for solving ODEs)
    mesh_path = 'meshes/remarked_mesh/mesh.xdmf'
    mesh, ct, ft = read_mesh(mesh_path)

    # Subdomain tags (same as is mesh). NB! ECS tag must always be zero.
    ECS_tag = 0
    neuron_tag = 1

    # Extract sub-meshes
    mesh_sub_0, sub_to_parent_0, sub_vertex_to_parent_0, _, _ = scifem.extract_submesh(mesh, ct, ECS_tag)
    mesh_sub_1, sub_to_parent_1, sub_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ct, neuron_tag)
    mesh_mem_1, mem_to_parent_1, mem_vertex_to_parent_1, _, _ = scifem.extract_submesh(mesh, ft, neuron_tag)

    # Create subdomains (extracellular space and cells)
    ECS = {"tag":ECS_tag,
           "name":"ECS",
           "mesh_sub":mesh_sub_0,
           "sub_to_parent":sub_to_parent_0,
           "sub_vertex_to_parent":sub_vertex_to_parent_0}

    neuron = {"tag":neuron_tag,
              "name":"neuron",
              "mesh_sub":mesh_sub_1,
              "sub_to_parent":sub_to_parent_1,
              "sub_vertex_to_parent":sub_vertex_to_parent_1,
              "mesh_mem":mesh_mem_1,
              "mem_to_parent":mem_to_parent_1}

    subdomain_list = [ECS, neuron]

    # Time variables
    t = dolfinx.fem.Constant(mesh, 0.0) # time constant

    dt = 0.1                            # global time step (ms)
    Tstop = 10                          # global end time (ms)
    n_steps_ODE = 25                    # number of ODE steps

    # Physical parameters
    C_M = 2.0                           # capacitance
    temperature = 300.0e3               # temperature (mK)
    F = 96485.0e3                       # Faraday's constant (mC/mol)
    R = 8.314e3                         # Gas Constant (mJ/(K*mol))
    D_Na = 1.33e-8                      # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-8                       # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-8                      # diffusion coefficients Cl (cm/ms)
    psi = F / (R * temperature)         # shorthand
    C_phi = C_M / dt                    # shorthand

    # Initial values
    Na_i_init = 12.838513108648856      # Intracellular Na concentration
    Na_e_init = 100.71925900027354      # extracellular Na concentration
    K_i_init = 124.15397583491901       # intracellular K concentration
    K_e_init = 3.3236967382705265       # extracellular K concentration
    Cl_e_init = Na_e_init + K_e_init    # extracellular CL concentration
    Cl_i_init = Na_i_init + K_i_init    # intracellular CL concentration

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

    # diffusion coefficients for each sub-domain
    D_Na = {0:dolfinx.fem.Constant(mesh_sub_0, D_Na),
            1:dolfinx.fem.Constant(mesh_sub_1, D_Na)}

    D_K = {0:dolfinx.fem.Constant(mesh_sub_0, D_K),
           1:dolfinx.fem.Constant(mesh_sub_1, D_K)}

    D_Cl = {0:dolfinx.fem.Constant(mesh_sub_0, D_Cl),
            1:dolfinx.fem.Constant(mesh_sub_1, D_Cl)}

    # initial concentrations for each sub-domain
    Na_init = {0:dolfinx.fem.Constant(mesh_sub_0, Na_e_init),
               1:dolfinx.fem.Constant(mesh_sub_1, Na_i_init)}

    K_init = {0:dolfinx.fem.Constant(mesh_sub_0, K_e_init),
              1:dolfinx.fem.Constant(mesh_sub_1, K_i_init)}

    Cl_init = {0:dolfinx.fem.Constant(mesh_sub_0, Cl_e_init),
               1:dolfinx.fem.Constant(mesh_sub_1, Cl_i_init)}

    # set source terms to be zero for all ion species
    f_source_Na = dolfinx.fem.Constant(mesh_sub_0, 0.0)
    f_source_K = dolfinx.fem.Constant(mesh_sub_0, 0.0)
    f_source_Cl = dolfinx.fem.Constant(mesh_sub_0, 0.0)

    # Create ions (channel conductivity is set below for each model)
    Na = {'c_init':Na_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z': 1.0,
          'name':'Na',
          'D':D_Na,
          'f_source':f_source_Na}

    K = {'c_init':K_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
         'z': 1.0,
         'name':'K',
         'D':D_K,
         'f_source':f_source_K}

    Cl = {'c_init':Cl_init,
          'bdry': dolfinx.fem.Constant(mesh_sub_0, (0.0, 0.0)),
          'z': -1.0,
          'name':'Cl',
          'D':D_Cl,
          'f_source':f_source_Cl}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    phi, phi_M_prev = create_functions_emi(subdomain_list, degree=1)
    c, c_prev = create_functions_knp(subdomain_list, ion_list, degree=1)

    # Set initial conditions for PDE system
    set_initial_conditions(ion_list, subdomain_list, c_prev)

    # Create new cell marker on gamma mesh
    cell_marker = 1
    cell_map_g = mesh_mem_1.topology.index_map(mesh_mem_1.topology.dim)
    num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts

    # Transfer mesh tags from ct to tags for gamma mesh on interface
    ct_g_1, _ = scifem.transfer_meshtags_to_submesh(
            ft, mesh_mem_1, mem_vertex_to_parent_1, mem_to_parent_1
    )

    #ct_g = dolfinx.mesh.meshtags(
        #mesh_mem_1, mesh_mem_1.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    #)

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models_neuron = {1: mm_hh}
    ct_g = {1: ct_g_1}
    Q = {1: phi_M_prev[neuron_tag].function_space}

    # Membrane parameters
    g_syn_bar = 0.5                     # synaptic conductivity (mS/cm**2)
    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 20e-4)

    # Set membrane parameters
    stim_params = {'g_syn_bar':g_syn_bar, 'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    mem_models_neuron = setup_membrane_model(
            stim_params, physical_parameters, ode_models_neuron, ct_g,
            Q, ion_list
    )

    # Add membrane model to neuron in subdomain list
    subdomain_list[1]['mem_models'] = mem_models_neuron

    # Create variational form emi problem
    a_emi, p_emi, L_emi = emi_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            mem_models_neuron, phi, phi_M_prev, c_prev, dt,
    )

    # Create variational form knp problem
    a_knp, p_knp, L_knp = knp_system(
            mesh, ct, ft, physical_parameters, ion_list, subdomain_list,
            mem_models_neuron, phi, phi_M_prev, c, c_prev, dt,
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [mem_to_parent_1, sub_to_parent_0, sub_to_parent_1]

    # Create solver emi problem
    problem_emi = create_solver_emi(
            a_emi, L_emi, phi, entity_maps, subdomain_list, comm
    )

    # Create solver knp problem
    problem_knp = create_solver_knp(
            a_knp, L_knp, c, entity_maps, subdomain_list, comm
    )

    # Crate dictionary for storing XDMF files and checkpoint filenames
    xdmf_sub = {}; xdmf_mem = {}
    fname_bp_sub = {}; fname_bp_mem = {}

    # Create files (XDMF and checkpoint) for saving results
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        xdmf = dolfinx.io.XDMFFile(comm, f"results/3D/results_sub_{tag}.xdmf", "w")
        xdmf.write_mesh(subdomain['mesh_sub'])
        adios4dolfinx.write_mesh(f"results/3D/checkpoint_sub_{tag}.bp", subdomain['mesh_sub'])
        xdmf_sub[tag] = xdmf
        fname_bp_sub[tag] = f"results/3D/checkpoint_sub_{tag}.bp"

        # Write membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
        if tag > 0:
            xdmf = dolfinx.io.XDMFFile(comm, f"results/3D/results_mem_{tag}.xdmf", "w")
            xdmf.write_mesh(subdomain['mesh_mem'])
            adios4dolfinx.write_mesh(f"results/3D/checkpoint_mem_{tag}.bp", subdomain['mesh_mem'])
            xdmf_mem[tag] = xdmf
            fname_bp_mem[tag] = f"results/3D/checkpoint_mem_{tag}.bp"

    for k in range(int(round(Tstop/float(dt)))):
        print(f'solving for t={float(t)}')

        # Solve ODEs
        solve_odes(
                mem_models_neuron, c_prev, phi_M_prev, ion_list, stim_params,
                dt, mesh, ct, subdomain_list, k
            )

        # Solve PDEs
        #problem_emi.solve()
        #problem_knp.solve()
        solve_emi(problem_emi)
        solve_knp(problem_knp)

        update_pde_variables(
                c, c_prev, phi, phi_M_prev, physical_parameters,
                ion_list, subdomain_list, mesh, ct,
        )

        # update time
        t.value = float(t + dt)

        # Write results to file
        for subdomain in subdomain_list:
            tag = subdomain['tag']
            # concentrations and potentials from previous time step to file
            write_to_file_sub(xdmf_sub[tag], fname_bp_sub[tag], tag, phi, c, ion_list, t)
            # membrane potential to file for all cellular subdomains (i.e. all subdomain but ECS)
            if tag > 0:
                write_to_file_mem(xdmf_mem[tag], fname_bp_mem[tag], tag, phi_M_prev, t)

    # Close XDMF files
    for subdomain in subdomain_list:
        tag = subdomain['tag']
        xdmf_sub[tag].close()
        if tag > 0:
            xdmf_mem[tag].close()

if __name__ == "__main__":
    solve_system()
