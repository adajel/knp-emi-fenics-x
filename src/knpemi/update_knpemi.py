from knpemidg.utils import plus, minus, pcws_constant_project
from dolfin import *

# define jump across the membrane (interface gamma)
JUMP = lambda f, n: minus(f, n) - plus(f, n)

def update_pde(c_prev, c, phi, n_g, Q, physical_params, ion_list,
        phi_M_prev_PDE, dt):

    # number of ions to solve for
    N_ions = len(ion_list[:-1])

    V = c.function_space()

    temperature = physical_params['temperature']
    F = physical_params['F']
    R = physical_params['R']
    rho = physical_params['rho']

    # update previous concentrations
    c_prev.assign(c)

    # update membrane potential
    phi_M_step_I = JUMP(phi, n_g)
    assign(phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, Q))

    # variable for eliminated ion concentration
    c_elim = 0

    # update Nernst potentials for next global time level
    for idx, ion in enumerate(ion_list[:-1]):
        # get current solution concentration
        c_k_ = split(c_prev)[idx]
        # update Nernst potential
        E = R * temperature / (F * ion['z']) * ln(plus(c_k_, n_g) / minus(c_k_, n_g))
        ion['E'].assign(pcws_constant_project(E, Q))

        # add ion specific contribution to eliminated ion concentration
        c_elim += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_k_

    # add contribution from background charge / immobile ions
    c_elim += - (1.0 / ion_list[-1]['z']) * rho

    # update eliminated ion concentration
    ion_list[-1]['c'].assign(project(c_elim, V.sub(N_ions - 1).collapse()))

    # update Nernst potential for eliminated ion
    E = R * temperature / (F * ion_list[-1]['z']) * ln(plus(ion_list[-1]['c'], n_g) / minus(ion_list[-1]['c'], n_g))
    ion_list[-1]['E'].assign(pcws_constant_project(E, Q))

    return
