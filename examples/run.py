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
)

interior_marker = 1
exterior_marker = 0

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"

comm = MPI.COMM_WORLD

def update_ode_variables(ode_model, c_prev, phi_M_prev, ion_list, meshes, k):
    """ Update parameters in ODE solver (based on previous PDEs step)
        specific to membrane model
    """
    c_e = c_prev['e']
    c_i = c_prev['i']

    # Get function space on membrane (gamma interface)
    Q = phi_M_prev.function_space

    # Get traces (in Q) of ECS and ICS concentrations
    K_e, K_i = interpolate_to_membrane(c_e[0], c_i[0], Q, meshes)
    Cl_e, Cl_i = interpolate_to_membrane(c_e[1], c_i[1], Q, meshes)
    Na_e, Na_i = interpolate_to_membrane(ion_list[-1]['c_e'], ion_list[-1]['c_i'], Q, meshes)

    # Set traces of ECS and ICS concentrations in ODE solver
    ode_model.set_parameter('K_e', K_e)
    ode_model.set_parameter('K_i', K_i)
    ode_model.set_parameter('Na_e', Na_e)
    ode_model.set_parameter('Na_i', Na_i)
    ode_model.set_parameter('Cl_e', Cl_e)
    ode_model.set_parameter('Cl_i', Cl_i)

    if k == 0:
        # If first time step do nothing, let initial value for phi_M
        # in ODE system be taken from ODE file
        pass
    else:
        # For all other time steps, update the membrane potential in ODE solver
        # based on previous PDEs time step
        ode_model.set_membrane_potential(phi_M_prev)

    return

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


def solve_odes(mem_models, c_prev, phi_M_prev, ion_list, stim_params, dt,
        meshes, k):
    # Solve ODEs (membrane models) for each membrane tag
    for mem_model in mem_models:

        # Update ODE variables based on PDEs output
        ode_model = mem_model['ode']
        update_ode_variables(
                ode_model, c_prev, phi_M_prev, ion_list, meshes, k
        )

        # Solve ODEs
        ode_model.step_lsoda(
                dt=dt,
                stimulus=stim_params['stimulus'],
                stimulus_locator=stim_params['stimulus_locator']
        )

        # Update PDE variables based on ODE output
        ode_model.get_membrane_potential(phi_M_prev)
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
    mesh_path = 'meshes/2D/mesh_2.xdmf'
    mesh, ct, ft = read_mesh(mesh_path)

    # subdomain markers
    exterior_marker = 0; interior_marker = 1,

    # gamma markers
    interface_marker = 1

    mesh_i, i_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ct, interior_marker
    )

    mesh_e, e_to_parent, _, _, _ = scifem.extract_submesh(
            mesh, ct, exterior_marker
    )

    mesh_g, g_to_parent, g_vertex_to_parent, _, _ = scifem.extract_submesh(
            mesh, ft, interface_marker
    )

    meshes = {"mesh":mesh, "mesh_e":mesh_e, "mesh_i":mesh_i, "mesh_g":mesh_g,
              "ct":ct, "ft":ft, "e_to_parent":e_to_parent,
              "i_to_parent":i_to_parent, "g_to_parent":g_to_parent}

    # Time variables
    t = dolfinx.fem.Constant(mesh, 0.0) # time constant

    dt = 1.0e-4                         # global time step (ms)
    Tstop = 2.0e-3                      # global end time (ms)
    n_steps_ODE = 25                    # number of ODE steps

    # Physical parameters
    C_M = 0.02                          # capacitance
    temperature = 300.0                 # temperature (mK)
    F = 96485.0                         # Faraday's constant (mC/mol)
    R = 8.314                           # Gas Constant (mJ/(K*mol))
    D_Na = 1.33e-9                      # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-9                       # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-9                      # diffusion coefficients Cl (cm/ms)
    psi = F / (R * temperature)         # shorthand
    C_phi = C_M / dt                    # shorthand

    """
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
    """

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
          'z': 1.0,
          'name':'Na',
          'D':D_Na,
          'f_source':f_source_Na}

    K = {'c_init':K_init,
          'bdry': dolfinx.fem.Constant(mesh_e, (0.0, 0.0)),
         'z': 1.0,
         'name':'K',
         'D':D_K,
         'f_source':f_source_K}

    Cl = {'c_init':Cl_init,
          'bdry': dolfinx.fem.Constant(mesh_e, (0.0, 0.0)),
          'z': -1.0,
          'name':'Cl',
          'D':D_Cl,
          'f_source':f_source_Cl}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_syn_bar = 0                     # synaptic conductivity (S/m**2)

    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    #stimulus_locator = lambda x: (x[0] < 20e-6)
    stimulus_locator = lambda x: True

    # Set membrane parameters
    stim_params = {'g_syn_bar':g_syn_bar, 'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_hh}

    phi, phi_M_prev = create_functions_emi(meshes, degree=1)
    c, c_prev = create_functions_knp(meshes, ion_list, degree=1)

    # Set initial conditions for PDE system
    set_initial_conditions(ion_list, c_prev)

    # Create new cell marker on gamma mesh
    cell_marker = 1
    cell_map_g = mesh_g.topology.index_map(mesh_g.topology.dim)
    num_cells_local = cell_map_g.size_local + cell_map_g.num_ghosts

    # Transfer mesh tags from ct to tags for gamma mesh on interface
    ct_g, _ = scifem.transfer_meshtags_to_submesh(
            ft, mesh_g, g_vertex_to_parent, g_to_parent
    )

    #ct_g = dolfinx.mesh.meshtags(
        #mesh_g, mesh_g.topology.dim, np.arange(num_cells_local, dtype=np.int32), cell_marker
    #)

    mem_models = setup_membrane_model(stim_params, physical_parameters,
            ode_models, ct_g, phi_M_prev.function_space, ion_list)

    # Create variational form emi problem
    a_emi, L_emi, dx = emi_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev, c_prev
    )

    # Create variational form knp problem
    a_knp, L_knp = knp_system(
            meshes, ct, ft, ct_g, physical_parameters, ion_list, mem_models,
            phi, phi_M_prev, c, c_prev, dt
    )

    # Specify entity maps for each sub-mesh to ensure correct assembly
    entity_maps = [g_to_parent, e_to_parent, i_to_parent]

    # Create solver emi problem
    problem_emi = create_solver_emi(a_emi, L_emi, phi, entity_maps, comm)
    # Create solver knp problem
    problem_knp = create_solver_knp(a_knp, L_knp, c, entity_maps, comm)

    l_I_Na = []
    l_I_K = []
    l_I_Cl = []
    l_phi_M = []
    l_I_sum = []
    l_psi = []

    l_m = []
    l_n = []
    l_h = []

    l_Na_i = []
    l_Na_e = []

    xdmf_e = dolfinx.io.XDMFFile(mesh.comm, "results/results_e.xdmf", "w")
    xdmf_i = dolfinx.io.XDMFFile(mesh.comm, "results/results_i.xdmf", "w")

    # write mesh and mesh tags to file
    xdmf_e.write_mesh(mesh_e)
    xdmf_i.write_mesh(mesh_i)

    #xdmf.write_meshtags(ct, mesh.geometry)
    # write initial conditions to file
    #write_to_file(xdmf, phi, c_prev, phi_M_prev, ion_list, 0)

    for k in range(int(round(Tstop/float(dt)))):
        print(f'solving for t={float(t)}')

        # Solve ODEs
        solve_odes(
                mem_models, c_prev, phi_M_prev, ion_list, stim_params,
                dt, meshes, k
        )

        # Solve PDEs
        problem_emi.solve()
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

        for mem_model in mem_models:

            Q = phi_M_prev.function_space
            ode_model = mem_model['ode']

            I_Na = dolfinx.fem.Function(Q)
            I_K = dolfinx.fem.Function(Q)
            I_Cl = dolfinx.fem.Function(Q)
            phi_M = dolfinx.fem.Function(Q)
            m = dolfinx.fem.Function(Q)
            n = dolfinx.fem.Function(Q)
            h = dolfinx.fem.Function(Q)

            Na_i_ODE = dolfinx.fem.Function(Q)
            Na_e_ODE = dolfinx.fem.Function(Q)
            psi_ODE = dolfinx.fem.Function(Q)

            ode_model.get_membrane_potential(phi_M)
            ode_model.get_parameter("I_ch_Na", I_Na)
            ode_model.get_parameter("I_ch_K", I_K)
            ode_model.get_parameter("I_ch_Cl", I_Cl)

            ode_model.get_state("m", m)
            ode_model.get_state("n", n)
            ode_model.get_state("h", h)

            ode_model.get_parameter("Na_i", Na_i_ODE)
            ode_model.get_parameter("Na_e", Na_e_ODE)
            ode_model.get_parameter("psi", psi_ODE)

            l_I_Na.append(I_Na.x.array[0]*1.0e3)
            l_I_K.append(I_K.x.array[0]*1.0e3)
            l_I_Cl.append(I_Cl.x.array[0]*1.0e3)
            l_phi_M.append(phi_M.x.array[0]*1.0e3)

            l_m.append(m.x.array[0])
            l_n.append(n.x.array[0])
            l_h.append(h.x.array[0])

            l_Na_i.append(Na_i_ODE.x.array[0])
            l_Na_e.append(Na_e_ODE.x.array[0])
            l_psi.append(psi_ODE.x.array[0])

            l_I_sum.append(I_Na.x.array[0] + I_K.x.array[0])

    xdmf_e.close()
    xdmf_i.close()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(l_I_Na, label="Na")
    plt.plot(l_I_K, label="K")
    plt.plot(l_I_Cl, label="Cl")
    plt.legend()
    plt.savefig("currents.png")
    plt.close()

    plt.figure()
    plt.plot(l_I_sum)
    plt.savefig("currents_sum.png")
    plt.close()

    plt.figure()
    plt.plot(l_phi_M)
    plt.savefig("pot.png")
    plt.close()

    plt.figure()
    plt.plot(l_psi)
    plt.savefig("psi.png")
    plt.close()

    plt.figure()
    plt.plot(l_Na_i)
    plt.plot(l_Na_e)
    plt.savefig("con.png")
    plt.close()

    plt.figure()
    plt.plot(l_m, label="m")
    plt.plot(l_n, label="n")
    plt.plot(l_h, label="h")
    plt.legend()
    plt.savefig("gats.png")
    plt.close()


if __name__ == "__main__":
    solve_system()
