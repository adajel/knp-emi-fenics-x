from dolfin import *
from petsc4py import PETSc

def create_solver_emi(direct, rtol, atol, threshold, lhs, rhs, prec, V):
    """ setup KSP solver for the emi sub-problem """

    # create solver
    ksp = PETSc.KSP().create()

    if direct:
        # set options for direct emi solver
        opts = PETSc.Options("EMI_DIR") # get options
        opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix
        opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix

        ksp.setOptionsPrefix("EMI_DIR")
        ksp.setFromOptions()            # update ksp with options set above
        pc = ksp.getPC()                # get pc
        pc.setType("lu")                # set solver to LU
        pc.setFactorSolverType("mumps") # set LU solver to use mumps
    else:
        # set options for iterative emi solver
        opts = PETSc.Options('EMI_ITER')
        opts.setValue('ksp_type', 'cg')
        opts.setValue('ksp_monitor_true_residual', None)
        opts.setValue('ksp_error_if_not_converged', 1)
        opts.setValue('ksp_max_it', 1000)
        opts.setValue('ksp_converged_reason', None)
        opts.setValue('ksp_initial_guess_nonzero', 1)
        opts.setValue('ksp_view', None)
        opts.setValue('pc_type', 'hypre')

        # set tolerances
        opts.setValue('ksp_rtol', rtol)
        opts.setValue('ksp_atol', atol)

        if threshold is not None:
            opts.setValue('pc_hypre_boomeramg_strong_threshold', threshold)

        ksp.setOptionsPrefix('EMI_ITER')
        ksp.setConvergenceHistory()
        ksp.setFromOptions()

    # assemble system
    AA, bb = map(assemble, (lhs, rhs))
    BB = assemble(prec)

    AA = as_backend_type(AA)
    BB = as_backend_type(BB)
    bb = as_backend_type(bb)

    x, _ = AA.mat().createVecs()

    # get Null space of A
    z = interpolate(Constant(1), V).vector()
    Z_ = PETSc.NullSpace().create([as_backend_type(z).vec()])

    return AA, BB, bb, Z_, ksp, x

def solve_emi(lhs, rhs, prec, AA, BB, bb, Z_, direct,
    ksp, phi, x):
    """ solve emi system using either a direct or iterative solver """

    # reassemble matrices and vector
    assemble(lhs, AA)
    assemble(rhs, bb)
    assemble(prec, BB)

    # convert matrices and vector
    AA = AA.mat()
    BB = BB.mat()
    bb = bb.vec()

    # set Null space of A
    AA.setNearNullSpace(Z_)

    if direct:
        Z_.remove(bb)

    if direct:
        ksp.setOperators(AA, AA)
    else:
        ksp.setOperators(AA, BB)

    # solve emi system
    ksp.solve(bb, x) # solve

    #x.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    phi.vector().vec().array_w[:] = x.array_r[:]
    # make assign above work in parallel
    phi.vector().vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return


def create_solver_knp(A, L, direct, rtol, atol, threshold, c):
    """ setup KSP solver for KNP sub-problem """

    # create solver
    ksp = PETSc.KSP().create()

    if direct:
        # set options direct solver
        opts = PETSc.Options("KNP_DIR")  # get options
        opts["mat_mumps_icntl_4"] = 1    # set amount of info output
        opts["mat_mumps_icntl_14"] = 40  # set percentage of ???

        pc = ksp.getPC()                 # get pc
        pc.setType("lu")                 # set solver to LU
        pc.setFactorSolverType("mumps")  # set LU solver to use mumps
        ksp.setOptionsPrefix("KNP_DIR")
        ksp.setFromOptions()             # update ksp with options set above
    else:
        # set options iterative solver
        opts = PETSc.Options('KNP_ITER')
        opts.setValue('ksp_type', 'gmres')
        opts.setValue('ksp_min_it', 5)
        opts.setValue("ksp_max_it", 1000)
        opts.setValue('pc_type', 'hypre')
        opts.setValue("ksp_converged_reason", None)
        opts.setValue("ksp_initial_guess_nonzero", 1)
        opts.setValue("ksp_view", None)
        opts.setValue("ksp_monitor_true_residual", None)

        opts.setValue('ksp_rtol', rtol)
        opts.setValue('ksp_atol', atol)

        if threshold is not None:
            opts.setValue('pc_hypre_boomeramg_strong_threshold', threshold)

        ksp.setOptionsPrefix('KNP_ITER')
        ksp.setFromOptions()

    # assemble
    AA, bb = map(assemble, (A, L))
    AA = as_backend_type(AA)
    bb = as_backend_type(bb)

    x, _ = AA.mat().createVecs()
    x.axpy(1, as_backend_type(c.vector()).vec())

    return AA, bb, x, ksp

def solve_knp(A, L, AA, bb, x, direct, ksp, c):
    """ solve knp system """

    # reassemble matrices and vector
    assemble(A, AA)
    assemble(L, bb)

    # convert matrices and vector
    AA = AA.mat()
    bb = bb.vec()

    if direct:
        # set operators
        AA.convert(PETSc.Mat.Type.AIJ)
        ksp.setOperators(AA, AA)

        # solve knp system with direct solver
        ksp.solve(bb, x)   # solve

        #x_knp.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    else:
        # set operators
        ksp.setOperators(AA, AA)

        # solve the knp system with iterative solver
        ksp.solve(bb, x)   # solve system

        #x.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # assign new value to function c
    c.vector().vec().array_w[:] = x.array_r[:]
    # make assign above work in parallel
    c.vector().vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return
