from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp
from knpemi.initialize_knpemi import set_initial_conditions
from knpemi.initialize_membrane import setup_membrane_model

from knpemi.knpemiSolver import create_solver_emi
from knpemi.knpemiSolver import create_solver_knp
from knpemi.knpemiSolver import solve_emi
from knpemi.knpemiSolver import solve_knp

from knpemi.script import interpolate_to_submesh, compute_interface_data

import mm_hh as mm_hh

import dolfinx
import scifem
from mpi4py import MPI
import numpy as np

from ufl import (
        ln,
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

def update_ode(ode_model, c_prev, phi_M_prev_PDE, ion_list, k):
    """ Update parameters in ODE solver (based on previous PDEs step)
        specific to membrane model
    """
    c_e_prev = c_prev['e']
    c_i_prev = c_prev['i']

    # Get function space on membrane (gamma interface)
    Q = phi_M_prev_PDE.function_space

    # Set extracellular trace of K concentration in ODE solver
    #K_e = c_e_prev[0](e_res)
    #---
    K_e = dolfinx.fem.Function(Q)
    K_e.x.array[:] = 3.32
    #---
    ode_model.set_parameter('K_e', K_e)

    # Set intracellular trace of Na concentration in ODE solver
    #Na_i = ion_list[-1]['c_i'](i_res)
    #---
    Na_i = dolfinx.fem.Function(Q)
    Na_i.x.array[:] = 12.83
    #---
    ode_model.set_parameter('Na_i', Na_i)

    # Update Nernst potentials in ODE solver
    #for ion in ion_list:
        #ode_model.set_parameter(f"E_{ion['name']}", ion['E'])

    E_Na = dolfinx.fem.Function(Q)
    E_K = dolfinx.fem.Function(Q)
    E_Cl = dolfinx.fem.Function(Q)

    E_Na.x.array[:] = -0.05323236322443255
    E_K.x.array[:] = 0.09346115007798299
    E_Cl.x.array[:] = 0.007097802159265801
    #---
    ode_model.set_parameter(f"E_Na", E_Na)
    ode_model.set_parameter(f"E_K", E_K)
    ode_model.set_parameter(f"E_Cl", E_Cl)
    #---

    if k == 0:
        # If first time step do nothing, let initial value for phi_M
        # in ODE system be taken from ODE file
        pass
    else:
        # For all other time steps, update the membrane potential in ODE solver
        # based on previous PDEs time step
        ode_model.set_membrane_potential(phi_M_prev_PDE)

    return

def update_pde(c_prev, c, phi, phi_M_prev_PDE, physical_parameters, ion_list,
        interface_to_parent, ct):
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

    # Update membrane potential
    Q = phi_M_prev_PDE.function_space
    phi_i_gamma = dolfinx.fem.Function(Q)
    phi_e_gamma = dolfinx.fem.Function(Q)

    interface_integration_entities = compute_interface_data(ct, interface_to_parent)
    mapped_entities = interface_integration_entities.copy()

    # interpolate trace of phi_e to function space over gamma
    interpolate_to_submesh(
        phi_e(e_res), phi_e_gamma, np.arange(len(interface_to_parent), dtype=np.int32), mapped_entities[:, :2]
    )

    # interpolate trace of phi_i to function space over gamma
    interpolate_to_submesh(
        phi_i(i_res), phi_i_gamma, np.arange(len(interface_to_parent), dtype=np.int32), mapped_entities[:, :2]
    )

    Q = phi_M_prev_PDE.function_space
    phi_M_prev = dolfinx.fem.Function(Q)
    phi_M_prev.interpolate(phi_i_gamma - phi_e_gamma)
    phi_M_prev_PDE.x.array[:] = phi_M_prev.x.array[:]

    # Scatter
    phi_M_prev_PDE.x.scatter_forward()

    # add contribution from background charge / immobile ions to eliminated ion
    c_elim = - (1.0 / ion_list[-1]['z']) * rho

    # update Nernst potentials for next global time level
    for idx, ion in enumerate(ion_list[:-1]):
        # get current solution concentration
        c_k_ = split(c_prev)[idx]
        # update Nernst potential
        E = R * temperature / (F * ion['z']) * ln(plus(c_k_, n_g) / minus(c_k_, n_g))
        ion['E'].assign(pcws_constant_project(E, Q))

        # add ion specific contribution to eliminated ion concentration
        c_elim += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_k_

    # update eliminated ion concentration
    ion_list[-1]['c'].assign(project(c_elim, V.sub(N_ions - 1).collapse()))

    # update Nernst potential for eliminated ion
    E = R * temperature / (F * ion_list[-1]['z']) * ln(plus(ion_list[-1]['c'], n_g) / minus(ion_list[-1]['c'], n_g))
    ion_list[-1]['E'].assign(pcws_constant_project(E, Q))

    return

def read_mesh(mesh_file):

    # Set ghost mode
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xdmf:
        # Read mesh and cell tags
        mesh = xdmf.read_mesh(ghost_mode=ghost_mode)
        ct = xdmf.read_meshtags(mesh, name='cell_marker')

        # Create facet entities, facet-to-cell connectivity and cell-to-cell connectivity
        mesh.topology.create_entities(mesh.topology.dim-1)
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

        # Read facets
        ft = xdmf.read_meshtags(mesh, name='facet_marker')

    return mesh, ct, ft

def solve_ODEs(phi_M_prev_PDE, c_prev, ion_list, mem_models, stim_params, k, dt):
    # Get stimuli
    stimulus = stim_params['stimulus']
    stimulus_locator = stim_params['stimulus_locator']

    # Solve ODEs (membrane models) for each membrane tag
    for mem_model in mem_models:
        # Update parameters in ODE solver (based on previous PDEs step)
        ode_model = mem_model['ode']
        update_ode(ode_model, c_prev, phi_M_prev_PDE,ion_list, k)

        # Solve ODEs
        ode_model.step_lsoda(dt=dt, \
            stimulus=stimulus, stimulus_locator=stimulus_locator)

        # Update PDE functions based on ODE output
        ode_model.get_membrane_potential(phi_M_prev_PDE)

        for ion, I_ch_k in mem_model['I_ch_k'].items():
            # Update src terms for next PDE step based on ODE output
            ode_model.get_parameter("I_ch_" + ion, I_ch_k)

def solve_system():
    """ Solve system (PDEs and ODEs) """
    # Read mesh and create sub-meshes for extra and intracellular domains and
    # for cellular membranes / interfaces (for solving ODEs)
    mesh_path = 'meshes/2D/mesh_2.xdmf'
    mesh, ct, ft = read_mesh(mesh_path)

    # subdomain markers
    exterior_marker = 0; interior_marker = 1,

    # gamma markers
    interface_marker = 1

    mesh_i, interior_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ct, interior_marker
    )

    mesh_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ct, exterior_marker
    )

    mesh_g, interface_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ft, interface_marker
    )

    meshes = {"mesh":mesh, "mesh_e":mesh_e, "mesh_i":mesh_i, "mesh_g":mesh_g}

    # Time variables
    t = dolfinx.fem.Constant(mesh, 0.0) # time constant
    dt = 1.0e-4                         # global time step (s)
    Tstop = 1.0e-2                      # global end time (s)
    n_steps_ODE = 25                    # number of ODE steps

    # Physical parameters
    C_M = 0.02                          # capacitance
    temperature = 300.0                 # temperature (K)
    F = 96485.0                         # Faraday's constant (C/mol)
    R = 8.314                           # Gas Constant (J/(K*mol))
    D_Na = 1.33e-9                      # diffusion coefficients Na (m/s)
    D_K = 1.96e-9                       # diffusion coefficients K (m/s)
    D_Cl = 2.03e-9                      # diffusion coefficients Cl (m/s)
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

    # diffusion coefficients for each sub-domain
    D_Na = {0:dolfinx.fem.Constant(mesh_e, D_Na),
            1:dolfinx.fem.Constant(mesh_i, D_Na)}

    D_K = {0:dolfinx.fem.Constant(mesh_e, D_K),
           1:dolfinx.fem.Constant(mesh_i, D_K)}

    D_Cl = {0:dolfinx.fem.Constant(mesh_e, D_Cl),
            1:dolfinx.fem.Constant(mesh_i, D_Cl)}

    # initial concentrations for each sub-domain
    Na_init = {0:dolfinx.fem.Constant(mesh_e, Na_e_init),
              1:dolfinx.fem.Constant(mesh_i, Na_i_init)}

    K_init = {0:dolfinx.fem.Constant(mesh_e, K_e_init),
              1:dolfinx.fem.Constant(mesh_i, K_i_init)}

    Cl_init = {0:dolfinx.fem.Constant(mesh_e, Cl_e_init),
               1:dolfinx.fem.Constant(mesh_i, Cl_i_init)}

    # set source terms to be zero for all ion species
    f_source_Na = dolfinx.fem.Constant(mesh_e, 0.0)
    f_source_K = dolfinx.fem.Constant(mesh_e, 0.0)
    f_source_Cl = dolfinx.fem.Constant(mesh_e, 0.0)

    # Create ions (channel conductivity is set below for each model)
    Na = {'c_init':Na_init,
          'bdry': dolfinx.fem.Constant(mesh_e, (0.0, 0.0)),
          'z':1.0,
          'name':'Na',
          'D':D_Na,
          'f_source':f_source_Na}

    K = {'c_init':K_init,
          'bdry': dolfinx.fem.Constant(mesh_e, (0.0, 0.0)),
         'z':1.0,
         'name':'K',
         'D':D_K,
         'f_source':f_source_K}

    Cl = {'c_init':Cl_init,
          'bdry': dolfinx.fem.Constant(mesh_e, (0.0, 0.0)),
          'z':-1.0,
          'name':'Cl',
          'D':D_Cl,
          'f_source':f_source_Cl}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_syn_bar = 0#10                   # synaptic conductivity (S/m**2)

    # set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 20e-6)

    # Set membrane parameters
    stim_params = {'g_syn_bar':g_syn_bar, 'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_hh}

    # Set solver parameters EMI (True is direct, and False is iterate)
    direct_emi = False
    rtol_emi = 1E-5
    atol_emi = 1E-40
    threshold_emi = 0.9

    # Set solver parameters KNP (True is direct, and False is iterate)
    direct_knp = False
    rtol_knp = 1E-7
    atol_knp = 1E-40
    threshold_knp = 0.75

    phi, phi_M_prev_PDE = create_functions_emi(meshes, degree=1)
    c, c_prev = create_functions_knp(meshes, ion_list, degree=1)

    set_initial_conditions(ion_list, c_prev)

    # Create new cell marker on gamma mesh
    cell_marker = 1
    cell_map_g = mesh_g.topology.index_map(mesh_g.topology.dim)
    num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts
    ct_g = dolfinx.mesh.meshtags(
        mesh_g, mesh_g.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    )

    mem_models = setup_membrane_model(stim_params, physical_parameters,
            ode_models, ct_g, phi_M_prev_PDE.function_space, ion_list)

    # For each ODE model, set parameters from initial conditions in PDE model
    for mem_model in mem_models:
        ode_model = mem_model['ode']
        update_ode(ode_model, c_prev, phi_M_prev_PDE, ion_list, 0)

    # Create variational form emi problem
    a_emi, L_emi = emi_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev_PDE, c_prev
    )

    # Create variational form knp problem
    a_knp, L_knp = knp_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev_PDE, c, c_prev, dt
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [interface_to_parent, exterior_to_parent, interior_to_parent]

    # Set solver parameters for emi solver
    direct_emi = True
    rtol_emi = 1.0
    atol_emi = 1.0
    threshold_emi = 1.0
    # Create solver emi problem
    solver_options_emi = create_solver_emi(
            direct_emi, rtol_emi, atol_emi, threshold_emi
    )

    # Set solver parameters for knp solver
    direct_knp = True
    rtol_knp = 1.0
    atol_knp = 1.0
    threshold_knp = 1.0
    # Create solver knp problem
    direct_knp = True
    solver_options_knp = create_solver_knp(
            direct_knp, rtol_knp, atol_knp, threshold_knp
    )

    # initialize save files
    xdmf_filename = "results/results.xdmf"

    mem_pot_point = []

    # Create an XDMFFile object to save results
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf_filename, "w") as xdmf:
        xdmf.write_mesh(mesh)

        # Time loop for solving PDE/ODE system
        for k in range(int(round(Tstop/float(dt)))):
            # write results to file
            xdmf.write_function(phi['e'], t=float(t))
            xdmf.write_function(phi['i'], t=float(t))

            for idx in range(len(ion_list)):
                # Determine the function source based on the index
                is_last = (idx == len(ion_list) - 1)

                c_i = ion_list[-1]['c_i'] if is_last else c['i'][idx]
                c_e = ion_list[-1]['c_e'] if is_last else c['e'][idx]

                # Write the functions to file
                (xdmf.write_function(f, t=float(t)) for f in (c_i, c_e))

            # Solve system
            print(f'solving for t={float(t)}')

            # Solve ODEs
            #solve_ODEs(phi_M_prev_PDE, c_prev, ion_list, mem_models, stim_params, k, dt)

            #mem_pot_point.append(phi_M_prev_PDE.x.array[1]*1.0e3)

            # Solve PDEs
            solve_emi(phi, a_emi, L_emi, solver_options_emi, entity_maps)
            solve_knp(c, a_knp, L_knp, solver_options_knp, entity_maps)

            update_pde(
                    c_prev, c, phi, phi_M_prev_PDE, physical_parameters,
                    ion_list, interface_to_parent, ct
            )

            # update time
            t.value = float(t + dt)

    xdmf.close()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mem_pot_point)
    plt.show()

if __name__ == "__main__":
    solve_system()
