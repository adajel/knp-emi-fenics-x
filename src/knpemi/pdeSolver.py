import dolfinx
from petsc4py import PETSc

from ufl import (
    extract_blocks,
)

def create_solver_emi(a, L, phi, entity_maps, subdomain_list, comm,
        direct=True, p=None, bcs=None, atol = 1E-40, rtol = 1E-5,
        threshold=None):
    """ Solve emi system using either a direct or iterative solver """

    if direct:
        # Set options direct solver
        petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_monitor": None,
                "ksp_error_if_not_converged": True,
            }
    else:
        # Set options iterative solver
        petsc_options = {
                'ksp_type':'cg',
                'ksp_monitor_true_residual':None,
                'ksp_error_if_not_converged':1,
                'ksp_max_it':1000,
                'ksp_converged_reason':None,
                'ksp_initial_guess_nonzero':1,
                'ksp_view':None,
                'pc_type':'hypre',
                'ksp_rtol':rtol,
                'ksp_atol':atol,
            }
        # Set threshold (if specified)
        if threshold is not None:
            petsc_options['pc_hypre_boomeramg_strong_threshold'] = threshold

    # Flatten extra and intracellular potentials into list (order by subdomain,
    # e.g. [phi_e, phi_i])
    u = [phi[tag] for tag in subdomain_list]

    if direct:
        # Create problem direct solver
        problem = dolfinx.fem.petsc.LinearProblem(
                  extract_blocks(a),
                  extract_blocks(L),
                  u=u,
                  bcs=bcs,
                  petsc_options=petsc_options,
                  petsc_options_prefix="emi_direct_",
                  entity_maps=entity_maps,
        )
    else:
        # Create problem iterative solver
        problem = dolfinx.fem.petsc.LinearProblem(
                  extract_blocks(a),
                  extract_blocks(L),
                  P=extract_blocks(p),
                  u=u,
                  bcs=bcs,
                  petsc_options=petsc_options,
                  petsc_options_prefix="emi_iterative_",
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


def create_solver_knp(a, L, c, entity_maps, subdomain_list, comm,
        direct=True, p=None, bcs=None, atol = 1E-40, rtol = 1E-5,
        threshold=None):
    """ Setup solver for the knp sub-problem """

    if direct:
        # Set options direct solver
        petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_monitor": None,
                "ksp_error_if_not_converged": True,
        }
    else:
        # Set options iterative solver
        petsc_options = {
                "ksp_type": "gmres",
                "ksp_min_it": 5,
                "ksp_max_it": 1000,
                "pc_type": 'hypre',
                "ksp_converged_reason": None,
                "ksp_initial_guess_nonzero": 1,
                "ksp_view": None,
                "ksp_monitor_true_residual": None,
                "ksp_rtol": rtol,
                "ksp_atol": atol,
        }
        # Set threshold (if specified)
        if threshold is not None:
            petsc_options['pc_hypre_boomeramg_strong_threshold'] = threshold

    # Flatted extra and intracellular concentrations into list (ordered by
    # first subdomain, then ion species, e.g. [Na_e, Cl_e, Na_i, Cl_i])
    u = [val for tag in subdomain_list for val in c[tag]]

    if direct:
        # Create problem direct solver
        problem = dolfinx.fem.petsc.LinearProblem(
                extract_blocks(a),
                extract_blocks(L),
                u=u,
                petsc_options=petsc_options,
                petsc_options_prefix="knp_direct_",
                entity_maps=entity_maps,
        )
    else:
        # Create problem iterative solver
        problem = dolfinx.fem.petsc.LinearProblem(
                  extract_blocks(a),
                  extract_blocks(L),
                  P=extract_blocks(p),
                  u=u,
                  petsc_options=petsc_options,
                  petsc_options_prefix="knp_iterative_",
                  entity_maps=entity_maps,
        )

    return problem
