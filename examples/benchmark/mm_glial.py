# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    phi_M_init = -85.85765274084892

    init_values = np.array([phi_M_init], dtype=np.float64)

    # State indices and limit checker
    state_ind = dict([("V", 0)])

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind = state_ind[state_name]

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """

    # Membrane parameters
    g_leak_Na = 0.1         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 1.696       # K leak conductivity (mS/cm**2)
    g_leak_Cl = 0.05        # Cl leak conductivity (mS/cm**2)

    m_K = 1.5               # threshold ECS K (mol/m^3)
    m_Na = 10               # threshold ICS Na (mol/m^3)
    I_max = 10.75975        # max pump strength (muA/cm^2)

    K_e_init = 3.092970607490389
    K_i_init = 99.3100014897692

    # Set initial parameter values
    init_values = np.array([0, g_leak_Cl, \
                            g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0,
                            m_K, m_Na, I_max], dtype=np.float64)

    # Parameter indices and limit checker
    param_ind = dict([("psi", 0), ("g_leak_Cl", 1), \
                      ("g_leak_Na", 2), ("g_leak_K", 3), \
                      ("z_Na", 4), ("z_K", 5), ("z_Cl", 6), \
                      ("Cm", 7), ("stim_amplitude", 8), \
                      ("I_ch_Na", 9), ("I_ch_K", 10), ("I_ch_Cl", 11), \
                      ("K_e", 12), ("K_i", 13), \
                      ("Na_e", 14), ("Na_i", 15), \
                      ("Cl_e", 16), ("Cl_i", 17), \
                      ("m_K", 18), ("m_Na", 19), ("I_max", 20)])

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind = param_ind[param_name]

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("V", 0)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict([("psi", 0), ("g_leak_Cl", 1), \
                       ("g_leak_Na", 2), ("g_leak_K", 3), \
                       ("z_Na", 4), ("z_K", 5), ("z_Cl", 6), \
                       ("Cm", 7), ("stim_amplitude", 8), \
                       ("I_ch_Na", 9), ("I_ch_K", 10), ("I_ch_Cl", 11), \
                       ("K_e", 12), ("K_i", 13), \
                       ("Na_e", 14), ("Na_i", 15), \
                       ("Cl_e", 16), ("Cl_i", 17), \
                       ("m_K", 18), ("m_Na", 19), ("I_max", 20)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

@cfunc(lsoda_sig, nopython=True)
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Assign states
    #assert(len(states)) == 4

    # Assign parameters
    #assert(len(parameters)) == 11

    # # Init return args
    # if values is None:
    #     values = np.zeros((4,), dtype=np.float64)
    # else:
    #     assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Parameter indices and limit checker
    psi = parameters[0]
    g_leak_Cl = parameters[1]
    g_leak_Na = parameters[2]
    g_leak_K = parameters[3]
    z_Na = parameters[4]
    z_K = parameters[5]
    z_Cl = parameters[6]
    Cm = parameters[7]
    stim_amplitude = parameters[8]
    I_ch_Na = parameters[9]
    I_ch_K = parameters[10]
    I_ch_Cl = parameters[11]

    K_e = parameters[12]
    K_i = parameters[13]
    Na_e = parameters[14]
    Na_i = parameters[15]
    Cl_e = parameters[16]
    Cl_i = parameters[17]

    m_K = parameters[18]
    m_Na = parameters[19]
    I_max = parameters[20]

    E_Na = 1/psi * 1/z_K * math.log(Na_e/Na_i)
    E_K = 1/psi * 1/z_K * math.log(K_e/K_i)
    E_Cl = 1/psi * 1/z_Cl * math.log(Cl_e/Cl_i)

    K_e_init = 3.092970607490389
    K_i_init = 99.3100014897692

    i_pump = I_max \
           * (K_e / (K_e + m_K)) \
           * (Na_i ** (1.5) / (Na_i ** (1.5) + m_Na ** (1.5)))

    # set conductance
    E_K_init = 1 / psi * np.log(K_e_init/K_i_init)
    dphi = states[0] - E_K
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[0])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # define and return current
    i_Kir = g_leak_K * g_Kir * (states[0] - E_K)              # umol/(cm^2*ms)

    # Expressions for the Sodium channel component
    i_Na = g_leak_Na * (states[0] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = i_Kir - 2 * i_pump

    # Expressions for the Chloride channel component
    i_Cl = g_leak_Cl * (states[0] - E_Cl)

    # set I_ch_Na
    parameters[9] = 0#i_Na
    # set I_ch_K
    parameters[10] = 0#i_K
    # set I_ch_Cl
    parameters[11] = 0#i_Cl

    # update membrane potential
    values[0] = 0#(- i_K - i_Na - i_Cl)/Cm
