import dolfinx

from ufl import (
    extract_blocks,
)

def create_solver_emi(direct, rtol, atol, threshold):
    """ Setup solver for the emi sub-problem """
    if direct:
        petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_monitor": None,
                "ksp_error_if_not_converged": True,
        }
    else:
        print("Iterative solver not implemented")

    return petsc_options

def solve_emi(phi, a, L, petsc_options, entity_maps, bc=None):
    """ solve emi system using either a direct or iterative solver """
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

    problem.solve()

    return
