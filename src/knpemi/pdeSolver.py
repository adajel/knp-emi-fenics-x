import dolfinx
from petsc4py import PETSc

from ufl import (
    extract_blocks,
)

def create_solver_emi(a, L, phi, entity_maps, comm, bcs=None):
    """ solve emi system using either a direct or iterative solver """

    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_monitor": None,
            "ksp_error_if_not_converged": True,
        }

    # Extract extra and intracellular potentials
    phi_e = phi['e']
    phi_i = phi['i']

    # Extract extra and intracellular potentials
    problem = dolfinx.fem.petsc.LinearProblem(
              extract_blocks(a),
              extract_blocks(L),
              u=[phi_e, phi_i],
              bcs=bcs,
              petsc_options=petsc_options,
              petsc_options_prefix="emi_direct_",
              entity_maps=entity_maps,
    )

    # TODO, make copy to asses if nullspace
    #A.assemble()
    #assert nullspace.test(A)

    # If no Dirichlet conditions (i.e. pure Neumann problem), A is singular
    # and we need to inform the solver about the null-space
    if bcs is None:
        # Set nullspace
        A = problem.A
        nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
        A.setNullSpace(nullspace)

    return problem


def create_solver_knp(a, L, c, entity_maps, bcs=None):
    """ Setup solver for the knp sub-problem """

    petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_monitor": None,
                "ksp_error_if_not_converged": True,
    }

    # Extract extra and intracellular concentrations
    c_e = c['e']
    c_i = c['i']

    # Extract extra and intracellular potentials
    problem = dolfinx.fem.petsc.LinearProblem(
            extract_blocks(a),
            extract_blocks(L),
            u=c_e + c_i,
            petsc_options=petsc_options,
            petsc_options_prefix="knp_direct_",
            entity_maps=entity_maps,
    )

    return problem
