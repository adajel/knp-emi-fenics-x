import dolfinx
from petsc4py import PETSc
from mpi4py import MPI

from ufl import (
    extract_blocks,
    MixedFunctionSpace,
)


def create_solver_emi(a, L, phi, entity_maps, comm):
    """ solve emi system using either a direct or iterative solver """
    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_monitor": None,
            "ksp_error_if_not_converged": True,
        }

    phi_e = phi['e']
    phi_i = phi['i']

    # Extract extra and intracellular potentials
    problem = dolfinx.fem.petsc.LinearProblem(
            extract_blocks(a),
            extract_blocks(L),
            u=[phi_e, phi_i],
            petsc_options=petsc_options,
            petsc_options_prefix="emi_direct_",
            entity_maps=entity_maps,
    )

    # Extract assembled rhs and lhs of system
    #A.assemble()
    #assert nullspace.test(A)

    # Set nullspace
    A = problem.A
    nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
    A.setNullSpace(nullspace)

    return problem
