import dolfinx
import numpy as np
from numbalsoda import lsoda

class MembraneModel():
    '''ODE on membrane defined by tagged facet function'''
    def __init__(self, ode, ft, tag, Q):
        '''
        Facets where facet_f[facet] == tag are governed by this ode whose 
        source terms will be taken from V
        '''
        assert isinstance(tag, int)

        # get indices for given tag
        indices = ft.find(tag)
        self.indices = indices.flatten()

        # get location of degrees of freedom for indices for given tag
        self.dof_locations = Q.tabulate_dof_coordinates()[self.indices]

        # number of points where ODEs is to be solved
        nodes = len(self.indices)
        self.nodes = nodes

        # get states and parameters for the ODEs
        self.states = np.array([ode.init_state_values() for _ in range(nodes)])
        self.parameters = np.array([ode.init_parameter_values() for _ in range(nodes)])

        self.tag = tag
        self.ode = ode
        self.prefix = ode.__name__
        self.time = 0

        print(f'\t{self.prefix} Number of ODE points on the membrane {nodes}')

    # --- Setting ODE state/parameter based on a FEM function
    def set_state(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__set_ODE('state', which, u, locator=locator)

    def set_parameter(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__set_ODE('parameter', which, u, locator=locator)

    # --- Getting PDE state/parameter based on a FEM function
    def get_state(self, which, u, locator=None):
        '''Set PDE function `u` based on ODE'''
        return self.__get_PDE('state', which, u, locator=locator)

    def get_parameter(self, which, u, locator=None):
        '''Set ODE based on PDE function `u`'''
        return self.__get_PDE('parameter', which, u, locator=locator)

    # --- Setting ODE states/parameters to "constant" values at certain locations
    def set_state_values(self, value_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        return self.__set_ODE_values('state', value_dict, locator=locator)

    def set_parameter_values(self, value_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        return self.__set_ODE_values('parameter', value_dict, locator=locator)

    # --- Convenience
    def set_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        return self.set_state('V', u, locator=locator)

    def get_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        return self.get_state('V', u, locator=locator)

    @property
    def V_index(self):
        return self.ode.state_indices('V')

    # ---- ODE integration ------
    def step_lsoda(self, dt, stimulus, stimulus_locator=None):
        '''Solve the ODEs forward by dt with optional stimulus'''
        if stimulus is None: stimulus = {}

        ode_rhs_address = self.ode.rhs_numba.address

        if stimulus_locator is None:
            stimulus_locator = lambda x: True
        stimulus_mask = np.fromiter(map(stimulus_locator, self.dof_locations), dtype=bool)

        print(f'\t{self.prefix} Stepping {self.nodes} ODEs')

        timer = dolfinx.common.Timer('ODE step LSODA')
        timer.start()
        tsteps = np.array([self.time, self.time+dt])
        for row, is_stimulated in enumerate(stimulus_mask):  # Local 
            row_parameters = self.parameters[row]

            if is_stimulated:
                for key, value in stimulus.items():
                    row_parameters[self.ode.parameter_indices(key)] = value

            current_state = self.states[row]

            new_state, success = lsoda(ode_rhs_address,
                                       current_state,
                                       tsteps,
                                       data=row_parameters,
                                       rtol=1.0e-8, atol=1.0e-10)
            assert success
            self.states[row, :] = new_state[-1]
        self.time = tsteps[-1]
        dt = timer.stop()
        print(f'\t{self.prefix} Stepped {self.nodes} ODES in {dt}s')

        return self.states

    # --- Work horses
    def __set_ODE(self, what, which, u, locator=None):
        '''ODE setting '''
        (get_index, destination) = {
            'state': (self.ode.state_indices, self.states),
            'parameter': (self.ode.parameter_indices, self.parameters)
        }[what]
        the_index = get_index(which)

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        source = u.x.array[:]
        if len(lidx) > 0:
            destination[lidx, the_index] = source[self.indices[lidx]]
        return self.states

    def __get_PDE(self, what, which, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        (get_index, source) = {
            'state': (self.ode.state_indices, self.states),
            'parameter': (self.ode.parameter_indices, self.parameters)
        }[what]
        the_index = get_index(which)

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        destination = u.x.array[:]

        if len(lidx) > 0:
            destination[self.indices[lidx]] = source[lidx, the_index]

        u.x.array[:] = destination

        return u

    def __set_ODE_values(self, what, value_dict, locator=None):
        '''Batch setter'''
        (destination, get_col) = {
            'state': (self.states, self.ode.state_indices),
            'parameter': (self.parameters, self.ode.parameter_indices)
        }[what]

        lidx = np.arange(self.nodes)
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]
        print(f'\t{self.prefix} Set {what} for {len(lidx)} ODES')

        if len(lidx) == 0: return destination

        coords = self.dof_locations[lidx]
        for param in value_dict:
            col = get_col(param)
            get_value = value_dict[param]
            for row, x in zip(lidx, coords):  # Counts the odes
                destination[row, col] = get_value(x)
        return destination


if __name__=="__main__":
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import scifem
    import mm_hh as ode

    comm = MPI.COMM_WORLD

    domain = mesh.create_unit_square(
        comm, 10, 10, ghost_mode=mesh.GhostMode.shared_facet
    )

    cell_map = domain.topology.index_map(domain.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    markers = np.full(num_cells_local, 1, dtype=np.int32)

    markers[
        dolfinx.mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] < 0.6)
    ] = 2

    ct = dolfinx.mesh.meshtags(
        domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32), markers
    )

    interface_facets = scifem.mesh.find_interface(ct, (1,), (2,))

    ft = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        interface_facets,
        np.full(interface_facets.shape, 1, dtype=np.int32),
    )
    ft.name = "facettag"

    gamma, gamma_to_parent, _, _, _ = scifem.mesh.extract_submesh(domain, ft, 1)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "parent.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ft, domain.geometry)
    xdmf.close()

    cell_map = gamma.topology.index_map(gamma.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    markers = np.full(num_cells_local, 1, dtype=np.int32)

    ct = dolfinx.mesh.meshtags(
        domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32), markers
    )

    Q = fem.functionspace(gamma, ("CG", 1))
    phi_M = fem.Function(Q)
    phi_M.x.array[:] = -74.38609374462003

    membrane = MembraneModel(ode, ct, 1, Q)

    #membrane.set_membrane_potential(phi_M)

    stimulus = {'stim_amplitude': 0.0,
                'stim_period': 0.0,
                'stim_duration': 0.0,
                'stim_start': 0}
    stimulus = None

    V_index = ode.state_indices('V')
    potential_history = []

    for _ in range(1000):
        membrane.step_lsoda(dt=0.0001, stimulus=stimulus)
        potential_history.append(1*membrane.states[:, V_index])
        #membrane.get_membrane_potential(phi_M)

    potential_history = np.array(potential_history)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(potential_history[:, 2])
    plt.savefig("plot_2.png")
