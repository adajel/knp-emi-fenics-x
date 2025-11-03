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
    phi_M_init = -0.07438609374462003     # membrane potential (mV)

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
    g_Na_bar = 1200         # Na max conductivity (mS/cm**2)
    g_K_bar = 360           # K max conductivity (mS/cm**2)
    g_leak_Na = 1.0         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 4.0         # K leak conductivity (mS/cm**2)

    m_K = 2                 # threshold ECS K (mol/cm**3)
    m_Na = 7.7              # threshold ICS Na (mol/cm**3)
    I_max = 0.449           # max pump strength (A/cm**2)

    I_ch_Na = 0
    I_ch_K = 0
    I_ch_Cl = 0

    stim_amplitude = 0

    # get values from PDE system
    z_Na = 0
    z_K = 0
    z_Cl = 0
    K_e = 0
    K_i = 0
    Na_e = 0
    Na_i = 0
    Cl_e = 0
    Cl_i = 0
    Cm = 0
    F = 0
    R = 0
    temperature = 0

    z_Na = 1
    z_K = 1
    z_Cl = -1

    # Initial values
    Na_i = 12.838513108648856       # Intracellular Na concentration
    Na_e = 100.71925900027354       # extracellular Na concentration
    K_i = 124.15397583491901        # intracellular K concentration
    K_e = 3.3236967382705265        # extracellular K concentration
    Cl_e = Na_e + K_e               # extracellular CL concentration
    Cl_i = Na_i + K_i               # intracellular CL concentration

    Cm = 0.02
    F = 96485
    R = 8.314
    temperature = 300

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar,
                            g_leak_Na, g_leak_K,
                            Cm, stim_amplitude,
                            I_ch_Na, I_ch_K, I_ch_Cl,
                            m_K, m_Na, I_max,
                            Na_e, Na_i,
                            K_e, K_i,
                            Cl_e, Cl_i,
                            z_Na, z_Cl, z_K,
                            F, R, temperature], dtype=np.float64)

    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                      ("g_leak_Na", 2), ("g_leak_K", 3),
                      ("Cm", 4), ("stim_amplitude", 5),
                      ("I_ch_Na", 6), ("I_ch_K", 7), ("I_ch_Cl", 8),
                      ("m_K", 9), ("m_Na", 10), ("I_max", 11),
                      ("Na_e", 12), ("Na_i", 13),
                      ("K_e", 14), ("K_i", 15),
                      ("Cl_e", 16), ("Cl_i", 17),
                      ("z_Na", 18), ("z_Cl", 19), ("z_K", 20),
                      ("F", 21), ("R", 22), ("temperature", 23)])

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

    param_inds = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                       ("g_leak_Na", 2), ("g_leak_K", 3),
                       ("Cm", 4), ("stim_amplitude", 5),
                       ("I_ch_Na", 6), ("I_ch_K", 7), ("I_ch_Cl", 8),
                       ("m_K", 9), ("m_Na", 10), ("I_max", 11),
                       ("Na_e", 12), ("Na_i", 13),
                       ("K_e", 14), ("K_i", 15),
                       ("Cl_e", 16), ("Cl_i", 17),
                       ("z_Na", 18), ("z_Cl", 19), ("z_K", 20),
                       ("F", 21), ("R", 22), ("temperature", 23)])

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
    Cm = parameters[4]
    stim_amplitude = parameters[5]
    I_ch_Na = parameters[6]
    I_ch_K = parameters[7]
    I_ch_Cl = parameters[8]
    m_K = parameters[9]
    m_Na = parameters[10]
    I_max = parameters[11]
    Na_e = parameters[12]
    Na_i = parameters[13]
    K_e = parameters[14]
    K_i = parameters[15]
    Cl_e = parameters[16]
    Cl_i = parameters[17]
    z_Na = parameters[18]
    z_Cl = parameters[19]
    z_K = parameters[20]
    F = parameters[21]
    R = parameters[22]
    temperature = parameters[23]

    #E_Na = R * temperature / (F * z_Na) * math.log(Na_i / Na_e)
    #E_K = R * temperature / (F * z_K) * math.log(K_i / K_e)
    #E_Cl = R * temperature / (F * z_Cl) * math.log(Cl_i / Cl_e)

    E_Na = -0.05323236322443255
    E_K = -0.09346115007798299
    E_Cl = 0.07097802159265801

    # Expressions for the m gate component
    alpha_m = 0.1e3 * (25. - 1.0e3*(states[3] + 65.0e-3))/(math.exp((25. - 1.0e3*(states[3] + 65.0e-3))/10.) - 1)
    beta_m = 4.e3*math.exp(- 1.0e3*(states[3] + 65.0e-3)/18.)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07e3*math.exp(- 1.0e3*(states[3] + 65.0e-3)/20.)
    beta_h = 1.e3/(math.exp((30.- 1.0e3*(states[3] + 65.0e-3))/10.) + 1)
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the n gate component
    alpha_n = 0.01e3*(10.- 1.0e3*(states[3] + 65.0e-3))/(math.exp((10.- 1.0e3*(states[3] + 65.0e-3))/10.) - 1.)
    beta_n = 0.125e3*math.exp(- 1.0e3*(states[3] + 65.0e-3) /80.)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    i_Stim = parameters[7] * np.exp(-np.mod(t, 0.03)/0.002)*(t < 125e-3)

    """
    # Expressions for the m gate component
    alpha_m = 0.1 * (25. - 1.0*(states[3] + 65.0))/(math.exp((25. - 1.0*(states[3] + 65.0))/10.) - 1)
    beta_m = 4.*math.exp(- 1.0*(states[3] + 65.0)/18.)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07*math.exp(- 1.0*(states[3] + 65.0)/20.)
    beta_h = 1./(math.exp((30.- 1.0*(states[3] + 65.0))/10.) + 1)
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the n gate component
    alpha_n = 0.01*(10.- 1.0*(states[3] + 65.0))/(math.exp((10.- 1.0*(states[3] + 65.0))/10.) - 1.)
    beta_n = 0.125*math.exp(- 1.0*(states[3] + 65.0) /80.)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Membrane component
    i_Stim = stim_amplitude * np.exp(-np.mod(t, 0.03)/0.002)*(t < 125)
    """

    i_pump = I_max / ((1 + m_K / K_e) ** 2 * (1 + m_Na / Na_i) ** 3)

    # Expressions for the Sodium channel component
    i_Na = (g_leak_Na + g_Na_bar * states[1] * math.pow(states[0], 3) + i_Stim) * \
           (states[3] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = (g_leak_K + g_K_bar * math.pow(states[2], 4)) * \
          (states[3] - E_K) - 2 * i_pump

    # set I_ch_Na
    #I_ch_Na = i_Na
    # set I_ch_K
    #I_ch_K = i_K
    # set I_ch_Cl
    #I_ch_Cl = 0.0

    parameters[6] = E_Na
    parameters[7] = E_K
    parameters[8] = E_Cl

    values[3] = (- i_K - i_Na)/Cm
