from knpemi.knpemiSolver import solve_emi, create_solver_emi
from knpemi.knpemiSolver import solve_knp, create_solver_knp
from knpemi.update_knpemi import update_pde

from knpemi.emiWeakForm import emi_system, create_functions_emi
from knpemi.knpWeakForm import knp_system, create_functions_knp
from knpemi.initialize_knpemi import initialize_params, set_initial_conditions
from knpemi.initialize_membrane import setup_membrane_model

from knpemidg.utils import interface_normal, plus, minus, pcws_constant_project

import mm_hh as mm_hh

from dolfin import *

def update_ode(ode_model, c_prev, n_g, Q, ion_list):
    """ Update parameters in ODE solver (based on previous PDEs step)
        specific to membrane model """

    # set extracellular trace of K concentration at membrane
    K_e = plus(c_prev.split()[0], n_g)
    ode_model.set_parameter('K_e', pcws_constant_project(K_e, Q))

    # set intracellular trace of Na concentration at membrane
    Na_i = minus(ion_list[-1]['c'], n_g)
    ode_model.set_parameter('Na_i', pcws_constant_project(Na_i, Q))

    return

def read_mesh(mesh_path):

    #mesh = Mesh()
    #infile = XDMFFile(mesh_path + 'mesh.xdmf')
    #infile.read(mesh)
    #cdim = mesh.topology().dim()
    #infile.close()

    ## convert mesh from nm to cm
    #mesh.coordinates()[:,:] *= 1e-7

    #subdomains = MeshFunction('size_t', mesh, mesh_path + 'subdomains.xml')
    #surfaces = MeshFunction('size_t', mesh, mesh_path + 'surfaces.xml')

    import os
    resolution = 2

    mesh_prefix = 'meshes/2D/'
    mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
    subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
    surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

    mesh = Mesh(mesh_path)
    subdomains = MeshFunction('size_t', mesh, subdomains_path)
    surfaces = MeshFunction('size_t', mesh, surfaces_path)

    return mesh, subdomains, surfaces


def solve_PDEs():
    # Step I: solve emi equations with known concentrations to obtain phi
    solve_emi()
    # Step II: solve knp equations with known phi to obtain concentrations
    solve_knp()
    # call updat function for updating PDE stuff

def solve_ODEs(mem_models, phi_M_prev_PDE, ion_list, k, c_prev, n_g, dt,
        stim_params):

    stimulus = stim_params['stimulus']
    stimulus_locator = stim_params['stimulus_locator']
    # solve ODEs (membrane models) for each membrane tag
    for mem_model in mem_models:

        ode_model = mem_model['ode']

        # Update membrane potential in ODE solver (based on previous
        # PDEs step) - except for the first step in the case where
        # phi_M_prev is given as constant when phi_M_init should
        # be the initial condition in the ODE system

        if k == 0:
            # first time step do nothing, let initial value for phi_M
            # in ODE system be taken from ODE file
            pass
        else:
            # update membrane potential in ODE solver based on previous PDEs
            # time step
            ode_model.set_membrane_potential(phi_M_prev_PDE)

        # Update Nernst potential in ODE solver (based on previous PDEs step)
        for i, ion in enumerate(ion_list):
            ode_model.set_parameter(f"E_{ion['name']}", ion['E'])

        # Update parameters in ODE solver (based on previous PDEs step) specific to membrane model
        update_ode(ode_model, c_prev, n_g, phi_M_prev_PDE.function_space(),
                ion_list)

        # Solve ODEs
        ode_model.step_lsoda(dt=dt, \
            stimulus=stimulus, stimulus_locator=stimulus_locator)

        # Update PDE functions based on ODE output
        ode_model.get_membrane_potential(phi_M_prev_PDE)

        # Update src terms for next PDE step based on ODE output
        for ion, I_ch_k in mem_model['I_ch_k'].items():
            # update src term for each ion species
            ode_model.get_parameter("I_ch_" + ion, I_ch_k)

def solve_system():
    """ Solve system (PDEs and ODEs) """

    # Time variables (PDEs)
    dt = 1.0e-4                      # global time step (s)
    Tstop = 1.0e-2                   # global end time (s)
    t = Constant(0.0)                # time constant

    # Time variables (ODEs)
    n_steps_ODE = 25                 # number of ODE steps

    # Physical parameters
    C_M = 0.02                       # capacitance
    temperature = 300                # temperature (K)
    F = 96485                        # Faraday's constant (C/mol)
    R = 8.314                        # Gas Constant (J/(K*mol))
    D_Na = Constant(1.33e-9)         # diffusion coefficients Na (m/s)
    D_K = Constant(1.96e-9)          # diffusion coefficients K (m/s)
    D_Cl = Constant(2.03e-9)         # diffusion coefficients Cl (m/s)
    psi = F / (R * temperature)      # shorthand
    C_phi = C_M / dt                 # shorthand

    # Initial values
    Na_i_init = 12.838513108648856   # Intracellular Na concentration
    Na_e_init = 100.71925900027354   # extracellular Na concentration
    K_i_init = 124.15397583491901    # intracellular K concentration
    K_e_init = 3.3236967382705265    # extracellular K concentration
    Cl_e_init = Na_e_init + K_e_init # extracellular CL concentration
    Cl_i_init = Na_i_init + K_i_init # intracellular CL concentration
    phi_M_init = Constant(-0.07438609374462003)   # membrane potential (V)
    phi_M_init_type = 'constant'

    # set background charge (no background charge in this scenario)
    rho_sub = {0:Constant(0), 1:Constant(0), 2:Constant(0)}

    # Set parameters
    physical_parameters = {'dt':dt, 'n_steps_ODE':dt, 'F':F, 'psi':psi,
            'phi_M_init':phi_M_init, 'C_phi':C_phi, 'C_M':C_M, 'R':R,
            'temperature':temperature,
            'rho_sub':rho_sub}

    # diffusion coefficients for each sub-domain
    D_Na_sub = {1:D_Na, 0:D_Na}
    D_K_sub = {1:D_K, 0:D_K}
    D_Cl_sub = {1:D_Cl, 0:D_Cl}

    # initial concentrations for each sub-domain
    Na_init_sub = {1:Constant(Na_i_init), 0:Constant(Na_e_init)}
    K_init_sub = {1:Constant(K_i_init), 0:Constant(K_e_init)}
    Cl_init_sub = {1:Constant(Cl_i_init), 0:Constant(Cl_e_init)}
    c_init_sub_type = 'constant'

    # set source terms to be zero for all ion species
    f_source_Na = Constant(0)
    f_source_K = Constant(0)
    f_source_Cl = Constant(0)

    # Create ions (channel conductivity is set below for each model)
    Na = {'c_init_sub':Na_init_sub,
          'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)),
          'z':1.0,
          'name':'Na',
          'D_sub':D_Na_sub,
          'f_source':f_source_Na}

    K = {'c_init_sub':K_init_sub,
         'c_init_sub_type':c_init_sub_type,
         'bdry': Constant((0, 0)),
         'z':1.0,
         'name':'K',
         'D_sub':D_K_sub,
         'f_source':f_source_K}

    Cl = {'c_init_sub':Cl_init_sub,
          'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)),
          'z':-1.0,
          'name':'Cl',
          'D_sub':D_Cl_sub,
          'f_source':f_source_Cl}

    # Create ion list. NB! The last ion in list will be eliminated
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_syn_bar = 10                   # synaptic conductivity (S/m**2)

    # set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 20e-6)

    # Set membrane parameters
    stim_params = {'g_syn_bar':g_syn_bar, 'stimulus':stimulus,
                   'stimulus_locator':stimulus_locator}

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_hh}

    mesh_path = 'meshes/2D/'
    mesh, subdomains, surfaces = read_mesh(mesh_path)

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

    # initialize parameters for variational form
    initialize_params(ion_list, physical_parameters, subdomains)

    phi, phi_M_prev_PDE = create_functions_emi(mesh, degree=1)
    c, c_prev = create_functions_knp(mesh, ion_list, degree=1)

    set_initial_conditions(ion_list, c_prev, subdomains)

    mem_models = setup_membrane_model(stim_params, physical_parameters,
            ode_models, surfaces, phi_M_prev_PDE.function_space(), ion_list)

    # create variational form emi problem
    lhs_emi, rhs_emi, prec_emi, n_g = emi_system(mesh,
                                                subdomains,
                                                surfaces,
                                                physical_parameters,
                                                ion_list,
                                                c_prev,
                                                phi, phi_M_prev_PDE, mem_models)

    # create variational form knp problem
    lhs_knp, rhs_knp = knp_system(mesh,
                                   subdomains,
                                   surfaces,
                                   physical_parameters,
                                   ion_list,
                                   mem_models,
                                   phi,
                                   phi_M_prev_PDE,
                                   dt,
                                   c, c_prev)


    AA_emi, BB_emi, bb_emi, Z_, ksp_emi, x_emi = create_solver_emi(direct_emi,
                                                                   rtol_emi,
                                                                   atol_emi,
                                                                   threshold_emi,
                                                                   lhs_emi,
                                                                   rhs_emi,
                                                                   prec_emi,
                                                                   phi.function_space())

    AA_knp, bb_knp, x_knp, ksp_knp = create_solver_knp(lhs_knp,
                                                       rhs_knp,
                                                       direct_knp,
                                                       rtol_knp,
                                                       atol_knp,
                                                       threshold_knp,
                                                       c)

    # initialize save files
    filename = "results/results.h5"
    h5_idx = 0
    h5_file = HDF5File(mesh.mpi_comm(), filename, 'w')
    h5_file.write(mesh, '/mesh')
    h5_file.write(subdomains, '/subdomains')
    h5_file.write(surfaces, '/surfaces')

    h5_file.write(c, '/concentrations',  h5_idx)
    h5_file.write(ion_list[-1]['c'], '/elim_concentration',  h5_idx)
    h5_file.write(phi, '/potential', h5_idx)

    for k in range(int(round(Tstop/float(dt)))):
        # solve system
        print(f'solving for t={float(t)}')

        solve_ODEs(mem_models, phi_M_prev_PDE, ion_list, k, c_prev, n_g, dt, stim_params)

        solve_emi(lhs_emi, rhs_emi, prec_emi, AA_emi, BB_emi, bb_emi, Z_, direct_emi, ksp_emi, phi, x_emi)
        solve_knp(lhs_knp, rhs_knp, AA_knp, bb_knp, x_knp, direct_knp, ksp_knp, c)
        update_pde(c_prev, c, phi, n_g, phi_M_prev_PDE.function_space(),
                physical_parameters, ion_list, phi_M_prev_PDE, dt)

        # update time
        t.assign(float(t + dt))

        # save results
        h5_idx += 1
        h5_file.write(c, '/concentrations',  h5_idx)
        h5_file.write(ion_list[-1]['c'], '/elim_concentration',  h5_idx)
        h5_file.write(phi, '/potential', h5_idx)

    # close file
    h5_file.close()

if __name__ == "__main__":
    solve_system()
