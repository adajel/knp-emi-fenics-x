import dolfinx

from ufl import (
    extract_blocks,
)

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
