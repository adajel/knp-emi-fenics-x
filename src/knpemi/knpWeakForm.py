from dolfin import *
from knpemidg.utils import interface_normal, plus, minus, pcws_constant_project

JUMP = lambda f, n: minus(f, n) - plus(f, n)

def create_measures(mesh, subdomains, surfaces):
    # define measures
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    return (dx, dS, ds)

def create_functions_knp(mesh, ion_list, degree=1):

    # number of ions to solve for
    N_ions = len(ion_list[:-1])

    # set up finite element space for concentrations (c)
    PK_knp = FiniteElement('DG', mesh.ufl_cell(), degree)
    ME = MixedElement([PK_knp]*N_ions)
    V = FunctionSpace(mesh, ME)

    # function for current and previous solution (concentrations) to knp
    c = Function(V)
    c_prev = Function(V)

    # initialize function for eliminated ion specific
    ion_list[-1]['c'] = interpolate(Constant(0), V.sub(N_ions - 1).collapse())

    return c, c_prev

def initialize(ion_list, mem_models, c_prev):
    """ calculate sum of alpha_sum and total ionic current """

    alpha_sum = 0

    for idx, ion in enumerate(ion_list):
        if idx == len(ion_list) - 1:
            # get eliminated concentrations from previous global step
            c_k_ = ion_list[-1]['c']
        else:
            # get concentrations from previous global step
            c_k_ = split(c_prev)[idx]

        # update alpha
        alpha_sum += ion['D'] * ion['z'] * ion['z'] * c_k_

    # sum of ion specific channel currents for each membrane tag
    I_ch = [0]*len(mem_models)

    # loop though membrane models to set total ionic current
    for jdx, mm in enumerate(mem_models):
        # loop through ion species
        for key, value in mm['I_ch_k'].items():
            # update total channel current for each tag
            I_ch[jdx] += mm['I_ch_k'][key]

    return alpha_sum, I_ch

def create_lhs_knp(us, vs, dx, dS, n, n_g, hA, tau_knp, phi, c_prev, ion_list,
        physical_parameters, dt):
    """ setup variational form for the knp system """

    # get psi
    psi = physical_parameters['psi']

    # initialize form
    a = 0

    for idx, ion in enumerate(ion_list[:-1]):
        # get trial and test functions
        u = us[idx]
        v = vs[idx]

        # get valence and diffusion coefficients
        z = ion['z']; D = ion['D']

        # upwinding: We first define function un returning:
        #       dot(u,n)    if dot(u, n) >  0
        #       0           if dot(u, n) <= 0
        #
        # We would like to upwind s.t.
        #   c('+') is chosen if dot(u, n('+')) > 0,
        #   c('-') is chosen if dot(u, n('+')) < 0.
        #
        # The expression:
        #       un('+')*c('+') - un('-')*c('-') = jump(un*c)
        #
        # give this. Given n('+') = -n('-'), we have that:
        #   dot(u, n('+')) > 0
        #   dot(u, n('-')) < 0.
        # As such, if dot(u, n('+')) > 0, we have that:
        #   un('+') is dot(u, n), and
        #   un('-') = 0.
        # and the expression above becomes un('+')*c('+') - 0*c('-') =
        # un('+')*c('+').

        # define upwind help function
        un = 0.5*(dot(D * grad(phi), n) + abs(dot(D * grad(phi), n)))

        # equation ion concentration diffusive term with SIP (symmetric)
        a += 1.0/dt * u * v * dx \
           + inner(D * grad(u), grad(v)) * dx \
           - inner(dot(avg(D * grad(u)), n('+')), jump(v)) * dS(0) \
           - inner(dot(avg(D * grad(v)), n('+')), jump(u)) * dS(0) \
           + tau_knp/avg(hA) * inner(jump(D * u), jump(v)) * dS(0)

        # drift (advection) terms + upwinding
        a += + z * psi * inner(D * u * grad(phi), grad(v)) * dx \
             - z * psi * jump(v) * jump(un * u) * dS(0)

    return a

def create_rhs_knp(vs, dx, dS, n, n_g, ion_list, c_prev, phi_M_prev_PDE,
                   alpha_sum, dt, mem_models, I_ch, physical_parameters, phi):
    """ setup variational form for the knp system """

    psi = physical_parameters['psi']        # combination of physical constants
    C_phi = physical_parameters['C_phi']    # physical parameters
    C_M = physical_parameters['C_M']        # membrane capacitance
    F = physical_parameters['F']            # Faraday's constant

    # initialize form
    L = 0

    for idx, ion in enumerate(ion_list[:-1]):
        # get trial and test functions
        v = vs[idx]

        # get previous concentration
        c_ = split(c_prev)[idx]

        # get valence and diffusion coefficients
        z = ion['z']; D = ion['D']

        # add terms for approximating time derivative
        L += 1.0/dt * c_ * v * dx
        # add src terms for ion injection ECS
        L += ion['f_source'] * v * dx(0)

        #if mms is None:
        # calculate alpha
        alpha = D * z * z * c_ / alpha_sum

        # calculate coupling coefficient
        C = alpha * C_M / (F * z * dt)

        # loop through each membrane model
        for jdx, mm in enumerate(mem_models):

            # get facet tag
            tag = mm['ode'].tag

            # robin condition with splitting
            g_robin_knp = phi_M_prev_PDE \
                        - dt / (C_M * alpha) * mm['I_ch_k'][ion['name']] \
                        + (dt / C_M) * I_ch[jdx]
            #else:
            # original robin condition (without splitting)
            #g_robin_knp = self.phi_M_prev_PDE \
            #            - self.dt / (C_M * alpha) * mm['I_ch_k'][ion['name']]

            # add coupling condition at interface
            L += JUMP(C * g_robin_knp * v, n_g) * dS(tag)

            # add coupling terms on interface gamma
            L += - jump(phi) * jump(C) * avg(v) * dS(tag) \
                 - jump(phi) * avg(C) * jump(v) * dS(tag)

    return L

def knp_system(mesh, subdomains, surfaces, physical_parameters, ion_list,
              mem_models, phi, phi_M_prev_PDE, dt, c,
              c_prev, degree=1):

    # number of ions to solve for
    N_ions = len(ion_list[:-1])

    # facet area and normal
    n = FacetNormal(mesh)
    hA = CellDiameter(mesh)
    n_g = interface_normal(subdomains, mesh)

    # DG penalty parameters
    gdim = mesh.geometry().dim()
    tau_knp = Constant(20*gdim*degree)

    dx, dS, ds = create_measures(mesh, subdomains, surfaces)

    V = c.function_space()

    us = TrialFunctions(V)
    vs = TestFunctions(V)

    # initialize variational formulation
    alpha_sum, I_ch = initialize(ion_list, mem_models, c_prev)

    # get left hand side knp system
    lhs = create_lhs_knp(us, vs,
                      dx, dS,
                      n, n_g,
                      hA, tau_knp,
                      phi, c_prev, ion_list, physical_parameters, dt)

    # get right hand side knp system
    rhs =  create_rhs_knp(vs,
                          dx, dS,
                          n, n_g,
                          ion_list,
                          c_prev, phi_M_prev_PDE,
                          alpha_sum, dt, mem_models,
                          I_ch, physical_parameters, phi)

    return lhs, rhs
