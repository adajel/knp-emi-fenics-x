import dolfinx
import numpy as np
import scifem

from knpemi.odeSolver import MembraneModel

def set_initial_conditions(ion_list, c_prev):
    """ Set initial conditions given by constants """

    for idx, ion in enumerate(ion_list):

        # Determine the target objects (c_e and c_i) based on the ion's index
        is_last = (idx == len(ion_list) - 1)
        c_e = ion_list[-1]['c_0'] if is_last else c_prev[0][idx]
        c_i = ion_list[-1]['c_1'] if is_last else c_prev[1][idx]

        # Apply initial conditions and scatter the data
        c_e.x.array[:] = ion['c_init'][0]
        c_e.x.scatter_forward()

        c_i.x.array[:] = ion['c_init'][1]
        c_i.x.scatter_forward()

    return

def setup_membrane_model(stim_params, physical_params, ode_models, ct, Q,
        ion_list):
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

    #mesh = meshes['mesh']
    #mesh_g = meshes['mesh_g']
    #e_to_parent = meshes['e_to_parent']
    #i_to_parent = meshes['i_to_parent']
    #g_to_parent = meshes['g_to_parent']
    #ct = meshes['ct']

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
    qe = dolfinx.fem.Function(Q, name="qe")
    qi = dolfinx.fem.Function(Q, name="qi")

    # Interpolate volume functions (on submesh) onto all cells of the interface submesh
    scifem.interpolation.interpolate_to_surface_submesh(
        ue, qe, np.arange(len(g_to_parent_map), dtype=np.int32), mapped_entities[:, :2]
    )
    qe.x.scatter_forward()
    scifem.interpolation.interpolate_to_surface_submesh(
        ui, qi, np.arange(len(g_to_parent_map), dtype=np.int32), mapped_entities[:, 2:]
    )
    qi.x.scatter_forward()

    # Compute the difference between the two interpolated functions
    #I = dolfinx.fem.Function(Q, name="i")
    #I.x.array[:] = qe.x.array - qi.x.array

    #reference = dolfinx.fem.Function(Q)
    #reference.interpolate(lambda x: fe(x) - fi(x))

    #qe_ref = dolfinx.fem.Function(Q)
    #qe_ref.interpolate(fe)
    #qi_ref = dolfinx.fem.Function(Q)
    #qi_ref.interpolate(fi)

    #np.testing.assert_allclose(qe.x.array, qe_ref.x.array)
    #np.testing.assert_allclose(qi.x.array, qi_ref.x.array)
    #np.testing.assert_allclose(I.x.array, reference.x.array, rtol=1e-14, atol=1e-14)

    # return extra and intracellular traces
    return qe, qi
