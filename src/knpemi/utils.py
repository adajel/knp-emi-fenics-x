import dolfinx
import numpy as np
import scifem

from knpemi.odeSolver import MembraneModel

from ufl import (
        ln,
)

i_res = "-"
e_res = "+"

def set_initial_conditions(ion_list, subdomain_list, c_prev):
    """ Set initial conditions given by constants """
    for tag, subdomain in subdomain_list.items():
        #tag = subdomain['tag']
        for idx, ion in enumerate(ion_list):
            # Determine the target objects (c_e and c_i) based on the ion's index
            is_last = (idx == len(ion_list) - 1)
            c_tag = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

            # Apply initial conditions and scatter the data
            c_tag.x.array[:] = ion['c_init'][tag]
            c_tag.x.scatter_forward()

    return


def setup_membrane_model(stim_params, physical_params, ode_models, ct, Q,
        ion_list):
    """ Initiate membrane model(s) containing membrane mechanisms (passive
        dynamics / ODEs) and src terms for PDE system """

    # Set membrane (ODE) stimuli parameters
    stimulus = stim_params["stimulus"]
    stimulus_locator = stim_params["stimulus_locator"]

    # List of membrane models
    mem_models = []

    # initialize and append ode models to list
    for tag_sub, ode in ode_models.items():

        # Initialize ODE model
        ode_model = MembraneModel(ode, ct, tag_sub, Q)

        # Set constants in ODE solver from PDE parameters (to ensure same values are used)
        ode_model.set_parameter_values({'Cm': lambda x: physical_params["C_M"]})
        ode_model.set_parameter_values({'psi': lambda x: physical_params["psi"]})
        # Set valence of ions in ODE solver
        for ion in ion_list:
            name = ion['name']
            ode_model.set_parameter_values({f'z_{name}': lambda x: ion['z']})

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

def interpolate_to_membrane(ue, ui, Q, mesh, ct, subdomain_list, tag):
    ECS = subdomain_list[0]
    ICS = subdomain_list[tag]

    mesh_g = ICS['mesh_mem']
    e_to_parent = ECS['sub_to_parent']
    i_to_parent = ICS['sub_to_parent']
    g_to_parent = ICS['mem_to_parent']

    num_facets_local = (
        mesh_g.topology.index_map(mesh_g.topology.dim).size_local
        + mesh_g.topology.index_map(mesh_g.topology.dim).num_ghosts
    )

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts

    g_to_parent_map = g_to_parent.sub_topology_to_topology(
        np.arange(num_facets_local, dtype=np.int32), inverse=False
    )

    # Compute ordered integration entities on the interface
    interface_integration_entities = scifem.compute_interface_data(
        ct, facet_indices=g_to_parent_map, include_ghosts=True
    )
    mapped_entities = interface_integration_entities.copy()

    # For each submesh, get the relevant integration entities
    parent_to_e = e_to_parent.sub_topology_to_topology(
        np.arange(num_cells_local, dtype=np.int32), inverse=True
    )
    parent_to_i = i_to_parent.sub_topology_to_topology(
        np.arange(num_cells_local, dtype=np.int32), inverse=True
    )
    mapped_entities[:, 0] = parent_to_e[interface_integration_entities[:, 0]]
    mapped_entities[:, 2] = parent_to_i[interface_integration_entities[:, 2]]
    assert np.all(mapped_entities[:, 0] >= 0)
    assert np.all(mapped_entities[:, 2] >= 0)

    # Create two functions on the interface submesh
    qe = dolfinx.fem.Function(Q, name=ue.name)
    qi = dolfinx.fem.Function(Q, name=ui.name)

    # Interpolate volume functions (on submesh) onto all cells of the interface submesh
    scifem.interpolation.interpolate_to_surface_submesh(
        ue, qe, np.arange(len(g_to_parent_map), dtype=np.int32), mapped_entities[:, :2]
    )
    qe.x.scatter_forward()
    scifem.interpolation.interpolate_to_surface_submesh(
        ui, qi, np.arange(len(g_to_parent_map), dtype=np.int32), mapped_entities[:, 2:]
    )
    qi.x.scatter_forward()

    # return extra and intracellular traces
    return qe, qi


def update_ode_variables(ode_model, c_prev, phi_M_prev, ion_list,
        subdomain_list, mesh, ct, tag, k):
    """ Update parameters in ODE solver (based on previous PDEs step) """
    # Get function space on membrane (gamma interface)
    Q = phi_M_prev.function_space

    # Set traces of ECS and ICS concentrations on membrane (from PDE solver) in ODE solver
    for idx, ion in enumerate(ion_list):
        # Determine the function source based on the index
        is_last = (idx == len(ion_list) - 1)
        c_e = ion_list[-1]['c_0'] if is_last else c_prev[0][idx]
        c_i = ion_list[-1][f'c_{tag}'] if is_last else c_prev[tag][idx]

        # Get and set extra and intracellular traces
        k_e, k_i = interpolate_to_membrane(c_e, c_i, Q, mesh, ct, subdomain_list, tag)
        ode_model.set_parameter(f"{ion['name']}_e", k_e)
        ode_model.set_parameter(f"{ion['name']}_i", k_i)

    # If first time step do nothing (the initial value for phi_M in ODEs are
    # taken from ODE file). For all other time steps, update the membrane
    # potential in ODE solver based on previous PDEs step
    if k > 0: ode_model.set_membrane_potential(phi_M_prev)

    return


def update_pde_variables(c, c_prev, phi, phi_M_prev, physical_parameters,
        ion_list, subdomain_list, mesh, ct):
    """ Update parameters in PDE solver for next time step """
    # Number of ions to solve for
    N_ions = len(ion_list[:-1])
    # Get physical parameters
    psi = physical_parameters['psi']
    rho = physical_parameters['rho']

    for tag, subdomain in subdomain_list.items():
        # Add contribution from immobile ions to eliminated ion
        c_elim_sum = - (1.0 / ion_list[-1]['z']) * rho['z'] * rho[tag]

        for idx, ion in enumerate(ion_list[:-1]):
            # Update previous concentration and scatter forward
            c_prev[tag][idx].x.array[:] = c[tag][idx].x.array
            c_prev[tag][idx].x.scatter_forward()

            # Add contribution to eliminated ion from ion concentration
            c_prev_sub = c_prev[tag][idx]
            c_elim_sum += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_prev_sub

        # Interpolate ufl sum onto V
        V = c[tag][tag].function_space
        c_elim = dolfinx.fem.Function(V)
        expr = dolfinx.fem.Expression(c_elim_sum, V.element.interpolation_points)
        c_elim.interpolate(expr)

        # Update eliminated ion concentration
        ion_list[-1][f'c_{tag}'].x.array[:] = c_elim.x.array

        # Update Nernst potentials for all cells (i.e. all subdomain but ECS)
        if tag != 0:
            # Update Nernst potentials for each ion we solve for
            for idx, ion in enumerate(ion_list[:-1]):
                # Get previous extra and intracellular concentrations
                c_e = c_prev[0][idx]
                c_i = c_prev[tag][idx]
                # Update Nernst potential
                ion['E'] =  1 / (psi * ion['z']) * ln(c_e(e_res) / c_i(i_res))

            # Update Nernst potential for eliminated ion
            c_e_elim = ion_list[-1][f'c_0']
            c_i_elim = ion_list[-1][f'c_{tag}']
            ion_list[-1]['E'] = 1 / (psi * ion['z']) * ln(c_e_elim(e_res) / c_i_elim(i_res))

            # Update membrane potential
            phi_e = phi[0]
            phi_i = phi[tag]
            phi_M_prev_sub = phi_M_prev[tag]

            # Update previous membrane potential (source term PDEs)
            Q = phi_M_prev_sub.function_space
            tr_phi_e, tr_phi_i = interpolate_to_membrane(phi_e, phi_i, Q, mesh, ct, subdomain_list, tag)
            phi_M_prev[tag].x.array[:] = tr_phi_i.x.array - tr_phi_e.x.array
            phi_M_prev[tag].x.scatter_forward()

    return
