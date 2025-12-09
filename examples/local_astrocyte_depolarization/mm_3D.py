# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original"
# model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    n_init = 0.1882020248041632         # gating variable n
    m_init = 0.016648440745822956       # gating variable m
    h_init = 0.8542015627820805         # gating variable h
    phi_M_init = -74.38609374462003   # membrane potential (V)

    init_values = np.array([m_init, h_init, n_init, phi_M_init], dtype=np.float64)

    # State indices and limit checker
    state_ind = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

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

    # Membrane parameters
    g_Na_bar = 120          # Na max conductivity (mS/cm**2)
    g_K_bar = 36            # K max conductivity (mS/cm**2)
    g_leak_Na = 0.1         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 0.4         # K leak conductivity (mS/cm**2)

    m_K = 2                 # threshold ECS K (mol/cm**3)
    m_Na = 7.7              # threshold ICS Na (mol/cm**3)
    I_max = 44.9            # max pump strength (A/cm**2)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar,
                            g_leak_Na, g_leak_K,
                            m_K, m_Na, I_max, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0,
                            0, 0, 0, 0], dtype=np.float64)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                      ("g_leak_Na", 2), ("g_leak_K", 3),
                      ("m_K", 4), ("m_Na", 5), ("I_max", 6),
                      ("Cm", 7), ("stim_amplitude", 8),
                      ("K_e", 9), ("K_i", 10),
                      ("Na_e", 11), ("Na_i", 12),
                      ("Cl_e", 13), ("Cl_i", 14),
                      ("I_ch_Na", 15), ("I_ch_K", 16), ("I_ch_Cl", 17),
                      ("z_Na", 18), ("z_K", 19), ("z_Cl", 20),
                      ("psi", 21)])

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
    state_inds = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

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

    # Parameter indices and limit checker
    param_inds = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                       ("g_leak_Na", 2), ("g_leak_K", 3),
                       ("m_K", 4), ("m_Na", 5), ("I_max", 6),
                       ("Cm", 7), ("stim_amplitude", 8),
                       ("K_e", 9), ("K_i", 10),
                       ("Na_e", 11), ("Na_i", 12),
                       ("Cl_e", 13), ("Cl_i", 14),
                       ("I_ch_Na", 15), ("I_ch_K", 16), ("I_ch_Cl", 17),
                       ("z_Na", 18), ("z_K", 19), ("z_Cl", 20),
                       ("psi", 21)])

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

    g_Na_bar = parameters[0]
    g_K_bar = parameters[1]
    g_leak_Na = parameters[2]
    g_leak_K = parameters[3]
    m_K = parameters[4]
    m_Na = parameters[5]
    I_max = parameters[6]
    Cm = parameters[7]
    stim_amplitude = parameters[8]
    K_e = parameters[9]
    K_i = parameters[10]
    Na_e = parameters[11]
    Na_i = parameters[12]
    Cl_e = parameters[13]
    Cl_i = parameters[14]
    I_ch_Na = parameters[15]
    I_ch_K = parameters[16]
    I_ch_Cl = parameters[17]
    z_Na = parameters[18]
    z_K = parameters[19]
    z_Cl = parameters[20]
    psi = parameters[21]

    E_Na = 1/psi * 1/z_K * math.log(Na_e/Na_i)
    E_K = 1/psi * 1/z_K * math.log(K_e/K_i)

    # Expressions for the m gate component
    alpha_m = 0.1 * (25. - (states[3] + 65.0))/(math.exp((25. - 1.0*(states[3] + 65.0))/10.) - 1)
    beta_m = 4.0 * math.exp(- (states[3] + 65.0)/18.)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07 * math.exp(- (states[3] + 65.0) / 20.)
    beta_h = 1.0 / (math.exp((30.0 - (states[3] + 65.0)) / 10.) + 1)
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the n gate component
    alpha_n = 0.01 * (10.0 - (states[3] + 65.0))/(math.exp((10.0 - (states[3] + 65.0))/10.) - 1.)
    beta_n = 0.125 * math.exp(- (states[3] + 65.0) / 80.0)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Membrane component
    i_Stim = stim_amplitude * np.exp(-np.mod(t, 30.0)/2.0)*(t < 125)

    i_pump = I_max / ((1 + m_K / K_e) ** 2 * (1 + m_Na / Na_i) ** 3)

    # Expressions for the Sodium channel component
    i_Na = (g_leak_Na + g_Na_bar * states[1] * math.pow(states[0], 3) + i_Stim) * \
           (states[3] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = (g_leak_K + g_K_bar * math.pow(states[2], 4)) * \
          (states[3] - E_K) - 2 * i_pump

    # set I_ch_Na
    parameters[15] = i_Na
    # set I_ch_K
    parameters[16] = i_K
    # set I_ch_Cl
    parameters[17] = 0.0

    values[3] = (- i_K - i_Na) / Cm
