from knpemidg.membrane import MembraneModel
from dolfin import *

# membrane model stuff
def setup_membrane_model(stim_params, physical_params, ode_models, surfaces, Q,
        ion_list):
    """
    Initiate membrane model(s) containing membrane mechanisms (passive
    dynamics / ODEs) and src terms for PDE system
    """

    # set membrane (ODE) stimuli parameters
    stimulus = stim_params["stimulus"]                 # stimulus
    stimulus_locator = stim_params["stimulus_locator"] # locator for stimulus

    # list of membrane models
    mem_models = []

    # initialize and append ode models to list
    for tag, ode in ode_models.items():

        # Initialize ODE model
        ode_model = MembraneModel(ode, facet_f=surfaces, tag=tag, V=Q)

        # Set ODE capacitance (to ensure same values are used)
        ode_model.set_parameter_values({'Cm': lambda x: physical_params["C_M"]})

        # dictionary for ion specific currents (i.e src terms PDEs)
        I_ch_k = {}
        # Initialize src terms for PDEs
        for ion in ion_list:
            # function for src term pde
            I_ch_k_ = Function(Q)
            # set src terms pde
            ode_model.get_parameter("I_ch_" + ion['name'], I_ch_k_)
            # set function in dictionary
            I_ch_k[ion['name']] = I_ch_k_

        # define membrane model (with ode model and src term for PDEs)
        mem_model = {'ode': ode_model, 'I_ch_k': I_ch_k}

        # append to list of membrane models
        mem_models.append(mem_model)

    return mem_models
