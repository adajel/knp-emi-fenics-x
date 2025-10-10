from knpemi.utils import make_global
from dolfin import *

def initialize_params(ion_list, physical_parameters, subdomains):
    """ initialize parameters """

    for idx, ion in enumerate(ion_list):
        # define global diffusion coefficients (for each ion)
        D = make_global(ion['D_sub'], subdomains)
        ion['D'] = D

    # define global function for background charge
    rho = make_global(physical_parameters["rho_sub"], subdomains)
    physical_parameters["rho"] = rho

    return

def set_initial_conditions(ion_list, c_prev, subdomains):
    """ set initial conditions given by constant """

    # get number of ions in system
    N_ions = len(ion_list[:-1])

    # get function-space
    V = c_prev.function_space()

    for idx, ion in enumerate(ion_list):
        # interpolate initial conditions to global function
        c_init = make_global(ion['c_init_sub'], subdomains)

        if idx == len(ion_list) - 1:
            # set initial concentrations for eliminated ion
            ion_list[-1]['c'].assign(interpolate(c_init, V.sub(N_ions - 1).collapse()))
        else:
            # set initial concentrations for other ions
            assign(c_prev.sub(idx), interpolate(c_init, V.sub(idx).collapse()))

    return
