# type imports
from typing import Callable

# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# pde imports
from petsc4py import PETSc
import ufl

# dolfinx imports
from dolfinx import fem

# local imports
from forward_solver import SteadyHeatForwardSolver2D
from adjoint_solver import AdjointSteadyHeatSolver2D

class TAOSolver2D:
    r"""
    Wrapper for PETSc.TAO quadratic solver to minimize the objective function:
        J[T(h), h] = 0.5 * [ \int_\Omega [T(h) - T_obs]^2/\sigma^2 + \alpha * \int_\Omega (\nabla h)^2 ],
    w.r.t. the unknown thermal conductivity h(x,y), subjected to the positivity bound h \geq h_min.
    """

    def __init__(self,
                 forward: SteadyHeatForwardSolver2D,
                 adjoint: AdjointSteadyHeatSolver2D,
                 *,
                 h_min: float | None = 1e-16,
                 h_max: float | None = None,
                 tao_type: str = "blmvm",
                 ls_algorithm : str = "armijo",
                 use_logh: bool = True,
                 gatol: float = 1e-6,
                 grtol: float = 1e-3,
                 gttol: float = 1e-3,
                 mit: int = 1000,
                 monitor: Callable | None = None,
                 options: bool = False,
                 verbose: int = 0,
    ):
        """
        Parameters
        ----------
        forward          : forward solver, SteadyHeatForwardSolver2D instance, must expose h, T, and solve()
        adjoint          : adjoint solver, AdjointSteadyHeatSolver2D instance, must expose alpha, sigma, and solve(), assemble_gradient()
        h_min            : positive lower bound for h
        h_max            : positive upper bound for h
        tao_type         : TAO solver type
        ls_algorithm     : TAO line search algorithm
        use_logh         : whether to optimize in log(h)
        gatol            : gradient absolute tolerance.
        grtol            : gradient relative tolerance.
        mit              : maximum number of iterations.
        monitor          : the monitor function to track the optimization progress.
        options          : whether and which options to configure the solver, refer to https://petsc.org/release/manualpages/Tao/TaoSetFromOptions for details.
        verbose          : verbosity level, 0-3
        """
        self.fwd = forward
        self.adj = adjoint
        self.alpha = adjoint.alpha
        self.sigma2 = adjoint.sigma2
        self.use_logh = use_logh
        self.verbose = verbose
        if self.use_logh:
            if h_min <= 0.0:
                raise ValueError("h_min must be positive to define log(h_min).")
            self.m = fem.Function(self.fwd.h.function.function_space)
            self.m.x.array[:] = np.log(np.clip(self.fwd.h.function.x.array, h_min, h_max))
            self.m.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                           mode=PETSc.ScatterMode.FORWARD)
            self.X0 = self.m.x.petsc_vec
            if self.verbose>0:
                if MPI.COMM_WORLD.rank == 0:
                    print("Optimize in m=log(h).")
                    with self.X0.localForm() as x_loc:
                        print("Initial guess for m=log(h) =", x_loc.array)
        else:
            self.X0 = self.fwd.h.function.x.petsc_vec
            if self.verbose>0:
                if MPI.COMM_WORLD.rank == 0:
                    print("Optimize in h.")
                    with self.X0.localForm() as x_loc:
                        print("Initial h=", x_loc.array)

        # Define the PETSc.TAO solver
        comm = self.X0.getComm()
        self.tao = PETSc.TAO().create(comm)

        # Set bounds on the thermal conductivity h(x,y)
        if self.use_logh:
            self._set_tao_bounds_on_logh(h_min=h_min, h_max=h_max)
        else:
            self._set_tao_bounds_on_h(h_min=h_min, h_max=h_max)

        # Set the TAO solver type and tolerances
        self.tao.setType(tao_type)  # "blmvm" handles simple bounds, only requires gradient
        self.tao.setTolerances(gatol=gatol, grtol=grtol, gttol=gttol)

        # Register the objective function and gradient callbacks
        self.tao.setObjectiveGradient(self._objgrad, None)

        # Set the maximum number of iterations
        self.tao.setMaximumIterations(mit)

        ## OPTIONAL: Monitor optimization progress
        if monitor is not None:
            self.tao.setMonitor(monitor)
        ## OPTIONAL: Command-line options
        if options:
            self.tao.setFromOptions()

        ## Hard-coded line search parameters
        ls = self.tao.getLineSearch()
        ls.setType(ls_algorithm)
        if self.verbose>=1:
            print("Line search algorithm:", ls.getType())
        if options:
            ls.setFromOptions()

    # Set bounds on m = log(h)
    def _set_tao_bounds_on_logh(self, *, h_min: float | None, h_max: float | None):
        """
        Set TAO bounds on m=log(h).
        """
        if self.verbose>0:
            print("Set bounds on m=log(h).")
        if h_min is not None and h_min <= 0.0:
            raise ValueError("h_min must be > 0 to define log(h_min).")

        m_lo = np.log(h_min)
        m_hi = PETSc.INFINITY if h_max is None else np.log(h_max)

        # Copy memory layout of X to get bound vectors with identical layout (including ghosts)
        self.lb = self.X0.duplicate()
        self.ub = self.X0.duplicate()
        # Set bound values
        self.lb.set(PETSc.ScalarType(m_lo))
        self.ub.set(PETSc.ScalarType(m_hi))

        # Set TAO variable bounds
        self.tao.setVariableBounds(self.lb, self.ub)

    # Set bounds on h
    def _set_tao_bounds_on_h(self, *, h_min: float | None, h_max: float | None):
        """
        Set TAO bounds on h.
        """
        if self.verbose>0:
            print("Set bounds on h.")
        h_lo = h_min if h_min is not None else 0.0
        h_hi =  PETSc.INFINITY if h_max is None else h_max

        # Copy memory layout of X to get bound vectors with identical layout (including ghosts)
        self.lb = self.X0.duplicate()
        self.ub = self.X0.duplicate()
        # Set bound values
        self.lb.set(PETSc.ScalarType(h_lo))
        self.ub.set(PETSc.ScalarType(h_hi))

        # Set TAO variable bounds
        self.tao.setVariableBounds(self.lb, self.ub)


    # Update h in forward and adjoint solvers from tao solver's current solution X
    def _update_h(self, X: PETSc.Vec):
        if self.use_logh:
            with self.fwd.h.function.x.petsc_vec.localForm() as h_loc, X.localForm() as x_loc:
                h_loc.array[:] = np.exp(x_loc.array)
            self.fwd.h.function.x.scatter_forward()
        else:
            with self.fwd.h.function.x.petsc_vec.localForm() as h_loc, X.localForm() as x_loc:
                h_loc.array[:] = x_loc.array
            self.fwd.h.function.x.scatter_forward()

    #  Objective and gradient callbacks
    def _objgrad(self, tao, X: PETSc.Vec, G: PETSc.Vec):
        """
        Assemble the gradient.
        Return the objective value J.
        Parameters
        ----------
        tao: PETSc.TAO
        X: the current thermal conductivity h(x,y) DOFs.
        G: the gradient vector dJ/dh.

        Returns
        -------
        J: the objective integrated over the domain.
        """
        # Copy the current X to the forward and adjoint solvers' thermal conductivity h(x,y)
        X.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

        self._update_h(X)

        # Forward solve to compute the temperature T
        self.fwd.solve()
        if self.verbose>=2:
            if MPI.COMM_WORLD.rank == 0:
                print("Current h = ", self.fwd.h.function.x.array)
        # Evaluate misfit and regularization terms
        ## Misfit
        misfit = ((self.fwd.T - self.adj.T_obs) ** 2) * ufl.dx
        misfit_scalar = fem.assemble_scalar(fem.form(misfit))

        ## Regularization
        regularization = self.alpha * ufl.inner(ufl.grad(self.fwd.h.function), ufl.grad(self.fwd.h.function)) * ufl.dx
        regularization_scalar = fem.assemble_scalar(fem.form(regularization))

        J = 0.5 * ( (misfit_scalar/self.sigma2) + regularization_scalar )
        if self.verbose>=2:
            if MPI.COMM_WORLD.rank == 0:
                print("Current J =", J)

        # Adjoint solve to compute the adjoint state lambda_L
        self.adj.solve()
        # Assemble the gradient vector
        grad_h = self.adj.assemble_gradient()  # PETSc Vec
        grad_h.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                mode=PETSc.ScatterMode.FORWARD)
        # Copy the gradient into the PETSc Vec G
        if self.use_logh:
            ## chain rule: dJ/dm = dJ/dh * dh/dm = dJ/dh * exp(m) = dJ/dh * h
            G.pointwiseMult(grad_h, self.fwd.h.function.x.petsc_vec)
        else:
            grad_h.copy(result=G)
        G.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
        if self.verbose==3:
            if MPI.COMM_WORLD.rank == 0:
                print("Current |G| =", G.norm())
        return J

    # Driver method to solve the optimization problem with the TAO solver
    def solve(self):
        """
        Solve for the thermal conductivity h(x,y) to minimize the objective function J.

        Returns
        -------
        PETSc.TAO convergence reason code, refer to https://petsc.org/release/manualpages/Tao/TaoConvergedReason/ for details.
        """
        self.tao.solve(x=self.X0)
        if self.verbose>0:
            print("Convergence Reason:", self.tao.getConvergedReason())
            print("For more details, refer to https://petsc.org/release/manualpages/Tao/TaoConvergedReason/")
        if self.use_logh:
            self.solution = np.exp(self.tao.getSolution().array)
        else:
            self.solution = self.tao.getSolution().array
        return self.solution
