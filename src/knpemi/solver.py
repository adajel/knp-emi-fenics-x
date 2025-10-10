from dolfin import *
import numpy as np
import sys
import os
import time
from petsc4py import PETSc

from knpemidg.utils import pcws_constant_project
from knpemidg.utils import interface_normal, plus, minus
from knpemidg.utils import CellCenterDistance
from knpemidg.membrane import MembraneModel

# define jump across the membrane (interface gamma)
JUMP = lambda f, n: minus(f, n) - plus(f, n)

parameters['ghost_mode'] = 'shared_facet'

# define colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# We here approximate the following system:
#     d(c_k)/dt + div(J_k) = 0, (knp)
#   - F sum_k z^k div(J_k) = 0, (emi)
#   where
#   J_k(c_k, phi) = - D grad(c_k) - z_k D_k psi c_k grad(phi)
#
#   We solve the system iteratively, by decoupling the first and second
#   equation, yielding the following system: Given c_k_ and phi_M_
#   iterate over the two following steps:
#
#       step I:  (emi) find phi by solving (2), with J^k(c_k_, phi)
#       step II: (knp) find c_k by solving (1) with J^k(c_k, phi), where phi
#                is the solution from step I
#      (step III: solve ODEs at interface, and update membrane potential)
#
# Membrane potential is defined as phi_i - phi_e, since we have
# marked cell in ECS with 0 and cells in ICS with 1 we have an
# interface normal pointing inwards (from lower to higher)
#    ____________________
#   |                    |
#   |      ________      |
#   |     |        |     |
#   | ECS |   ICS  |     |
#   |  0  |->  1   |     |
#   | (+) |   (-)  |     |
#   |     |________|     |
#   |                    |
#   |____________________|
#
# Normal will always point from lower to higher (e.g. from 0 -> 1)
# NB! The code assumes that all interior facets are tagged with 0.

self.ion_list = ion_list            # list of ions species
self.N_ions = len(ion_list[:-1])    # number of ions

def solve_system_passive(self, Tstop, t, solver_params,
        filename=None, save_fields=False, save_solver_stats=False):
    """
    Solve system with passive membrane mechanisms
    """

    # Set filename for saving results
    self.filename = filename
    self.save_fields = save_fields
    self.save_solver_stats = save_solver_stats

    # Setup solver and parameters
    self.solver_params = solver_params               # parameters for solvers

    self.direct_emi = solver_params.direct_emi       # choice of solver emi

    if not self.direct_emi:
        self.rtol_emi = solver_params.rtol_emi           # relative tolerance emi
        self.atol_emi = solver_params.atol_emi           # absolute tolerance emi
        self.threshold_emi = solver_params.threshold_emi # threshold emi

    self.direct_knp = solver_params.direct_knp       # choice of solver knp

    if not self.direct_knp:
        self.rtol_knp = solver_params.rtol_knp           # relative tolerance knp
        self.atol_knp = solver_params.atol_knp           # absolute tolerance knp
        self.threshold_knp = solver_params.threshold_knp # threshold knp

    self.splitting_scheme = False                    # no splitting scheme

    # Setup variational formulations
    self.setup_varform_emi()
    self.setup_varform_knp()

    # Setup solvers
    self.setup_solver_emi()
    self.setup_solver_knp()

    # If user has specified that solver statistics or fields should be
    # saved without specifying a filename raise error.
    if filename is None and (self.save_solver_stats or self.save_fields):
        print(f"Please specify filename when initiating \
               Solver.solve_system_passive() method: \n \
               solve_system_passive(self, Tstop, t, solver_params, filename=None, save_fields=False, save_solver_stats=False)")
        sys.exit(0)

    if self.save_fields:
        # initialize file for fields (solutions to PDEs)
        self.init_h5_savefile(filename + 'results.h5')

    if self.save_solver_stats:
        # initialize file for solver statistics (CPU timings, number of iterations etc)
        self.init_solver_stats(filename + 'solver/')

    # Solve system (PDEs and ODEs)
    for k in range(int(round(Tstop/float(self.dt)))):
        # Start timer (ODE solve)
        ts = time.perf_counter()

        # End timer (ODE solve)
        te = time.perf_counter()
        res = te - ts
        print(f"{bcolors.OKGREEN} CPU Execution time ODE solve: {res:.4f} seconds {bcolors.ENDC}")

        # Solve PDEs
        self.solve_for_time_step(k, t)
        #self.solve_for_time_step_picard(k, t)

        # Save results
        if (k % self.sf) == 0 and self.save_fields:
            self.save_h5()

    # Close files
    if self.save_fields:
        self.close_h5()
    if self.save_solver_stats:
        self.close_solver_stats()

    # combine solution for the potential and concentrations
    uh = split(self.c) + (self.phi,)

    return uh, self.ion_list[-1]['c']


    def update_ode(self, ode_model):
        """ Update parameters in ODE solver (based on previous PDE-step) that
            are specific to membrane model. This method is meant to be
            implemented by subclasses """

        raise NotImplementedError("Subclasses must implement the 'update_ode' function.")

        return

#------------------------------------------------------
# MMS stuff emi

# For the MMS problem, we need unique tags for each of the interface walls
if self.mms is not None:
    self.lm_tags = [1, 2, 3, 4]

for idx, ion in enumerate(self.ion_list):
        # define global coupling coefficient (for MMS case, for each ion)
        if self.mms is not None:
            C = self.make_global(ion['C_sub'])
            ion['C'] = C

    # add term in varform emi
    # add terms for manufactured solutions test
    if self.mms is not None:
        lm_tags = self.lm_tags
        g_robin_emi = self.mms.rhs['bdry']['u_phi']
        fphi1 = self.mms.rhs['volume_phi_1']
        fphi2 = self.mms.rhs['volume_phi_2']
        g_flux_cont = self.mms.rhs['bdry']['stress']
        phi1e = self.mms.solution['phi_1']
        phi2e = self.mms.solution['phi_2']

        # add robin condition at interface
        L += sum(C_phi * inner(g_robin_emi[tag], JUMP(v_phi, self.n_g)) * dS(tag) for tag in lm_tags)

        # add coupling term at interface
        a += sum(C_phi * inner(jump(u_phi), jump(v_phi))*dS(tag) for tag in lm_tags)

        # MMS specific: add source terms
        L += inner(fphi1, v_phi)*dx(1) \
           + inner(fphi2, v_phi)*dx(0) \

        # MMS specific: we don't have normal cont. of I_M across interface
        L += sum(inner(g_flux_cont[tag], plus(v_phi, n_g)) * dS(tag) for tag in lm_tags)

        # Neumann
        for idx, ion in enumerate(self.ion_list):
            # MMS specific: add neumann boundary terms (not zero in MMS case)
            L += - F * ion['z'] * dot(ion['bdry'], n) * v_phi * ds

# MMS stuff KNP
             # add terms for manufactured solutions test
            if self.mms is not None:
                # get mms data
                fc1 = ion['f1']
                fc2 = ion['f2']
                g_robin_knp_1 = ion['g_robin_1']
                g_robin_knp_2 = ion['g_robin_2']

                # get global coupling coefficients
                C = ion['C']; C_1 = ion['C_sub'][1]; C_2 = ion['C_sub'][0]

                lm_tags = self.lm_tags

                # MMS specific: add source terms
                L += inner(fc1, v_c)*dx(1) \
                   + inner(fc2, v_c)*dx(0) \

                # coupling terms on interface gamma
                L += - sum(jump(phi) * jump(C) * avg(v_c) * dS(tag) for tag in lm_tags) \
                     - sum(jump(phi) * avg(C) * jump(v_c) * dS(tag) for tag in lm_tags)

                # define robin condition on interface gamma
                L += sum(inner(C_1 * g_robin_knp_1[tag], minus(v_c, n_g)) * dS(tag) for tag in lm_tags) \
                   - sum(inner(C_2 * g_robin_knp_2[tag], plus(v_c, n_g)) * dS(tag) for tag in lm_tags)

                # MMS specific: add neumann contribution
                L += - dot(ion['bdry'], n) * v_c * ds


#------------------------------------------------------------ 

#----------------------------------------------------------


   # setup preconditioner EMI
    up, vp = TrialFunction(self.V_emi), TestFunction(self.V_emi)

    # scale mass matrix to get condition number independent from domain length
    mesh = self.mesh
    gdim = mesh.geometry().dim()

    for axis in range(gdim):
        x_min = mesh.coordinates().min(axis=0)
        x_max = mesh.coordinates().max(axis=0)

        x_min = np.array([MPI.min(mesh.mpi_comm(), xi) for xi in x_min])
        x_max = np.array([MPI.max(mesh.mpi_comm(), xi) for xi in x_max])

    # scaled mess matrix
    Lp = Constant(max(x_max - x_min))
    # self.B_emi is singular so we add (scaled) mass matrix
    mass = self.kappa*(1/Lp**2)*inner(up, vp)*dx

    B = a + mass

    # set forms lhs (A) and rhs (L)
    self.a_emi = a
    self.L_emi = L
    self.B_emi = B
    self.u_phi = u_phi

    return


