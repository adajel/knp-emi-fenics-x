import dolfinx

def set_initial_conditions(ion_list, c_prev):
    """ Set initial conditions given by constants """
    for idx, ion in enumerate(ion_list):
        # Determine the target objects (c_e and c_i) based on the ion's index
        is_last = (idx == len(ion_list) - 1)

        c_e = ion_list[-1]['c_e'] if is_last else c_prev['e'][idx]
        c_i = ion_list[-1]['c_i'] if is_last else c_prev['i'][idx]

        # Apply initial conditions and scatter the data
        c_e.x.array[:] = ion['c_init'][0]
        c_e.x.scatter_forward()

        c_i.x.array[:] = ion['c_init'][1]
        c_i.x.scatter_forward()

    return

def setup_membrane_model(stim_params, physical_params, ode_models, ct, Q, ion_list):
    """
    Initiate membrane model(s) containing membrane mechanisms (passive
    dynamics / ODEs) and src terms for PDE system
    """
    # Set membrane (ODE) stimuli parameters
    stimulus = stim_params["stimulus"]
    stimulus_locator = stim_params["stimulus_locator"]

    # List of membrane models
    mem_models = []

    # initialize and append ode models to list
    for tag, ode in ode_models.items():

        # Initialize ODE model
        ode_model = MembraneModel(ode, ct, tag, Q)

        # Set ODE capacitance (to ensure same values are used)
        ode_model.set_parameter_values({'Cm': lambda x: physical_params["C_M"]})

        # Dictionary for ion specific currents (i.e src terms PDEs)
        I_ch_k = {}
        # Initialize src terms for PDEs
        for ion in ion_list:
            # Function for src term pde
            I_ch_k_ = dolfinx.fem.Function(Q)
            # Set src terms pde from ode model
            ode_model.get_parameter("I_ch_" + ion['name'], I_ch_k_)
            # Set function in dictionary
            I_ch_k[ion['name']] = I_ch_k_

        # Define membrane model (with ode model and src term for PDEs)
        mem_model = {'ode': ode_model, 'I_ch_k': I_ch_k}
        # Append to list of membrane models
        mem_models.append(mem_model)

    return mem_models
