# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# pde imports
from petsc4py import PETSc
import ufl

# dolfinx imports
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

# local imports
from forward_solver import SteadyHeat2DForwardSolver

class SteadyHeat2DAdjointSolver(SteadyHeat2DForwardSolver):
    r"""
    Adjoint solver for the adjoint equation of the steady-state Poisson heat equation on a 2D unit square:
        \nabla\cdot(h \nabla\lambda_L) = - (\partial J / \partial T) = - \int_\Omega [T(h) - T_obs]/\sigma^2,
    where \int_\Omega = \sum_x \sum_y.
    J is the objective function:
        J[T(h), h] = 0.5 * [ \int_\Omega [T(h) - T_obs]^2/\sigma^2 + \alpha * \int_\Omega (\nabla h)^2 ],
    with \sigma^2 being the variance of the sensor/measurement Gaussian noise.
    The Tikhonov regularization term \alpha * \int_\Omega (\nabla h)^2 ensures smoothness of the solution h.
    \lambda_L is the Lagrangian multiplier:
        L = J + lambda_L^T*g[T(h), h],
    with g[T(h), h] being the residual of the steady-state Poisson heat equation:
        g[T(h), h] = -\nabla\cdot(h\nabla T) - q,
    Dirichlet boundary condition on the bottom wall: lambda_L(y=0)=0.
    Neumann boundary conditions on the other three insulated walls: \nabla\lambda_L\cdot\hat{n}=0.
    """
    def __init__(self,
                 forward_solver: SteadyHeat2DForwardSolver,
                 T_obs: fem.Function or array-like,
                 sigma: float = 1.0,
                 alpha: float = 0.0,
                 DBC_value: float = 0.0,
                 petsc_opts: dict | None = None):
        """
        Parameters
        ----------
        forward_solver  : forward solver instance, providing mesh, V, bottom_dofs, h, q, T.
        T_obs           : the observed/measured temperature.
        sigma           : std. of the sensor (Gaussian) noise.
        DBC_value       : Dirichlet BC for \lambda_L at the bottom, i.e. \lambda_L(y=0)=\lambda_{L,0}.
        petsc_opts      : Dictionary of PETSc options, copied from forward solver if None.
        """
        # Copy forward solver attributes:
        ## mesh, function space, botom_dofs;
        self.mesh, self.V, self.bottom_dofs = forward_solver.mesh, forward_solver.V, forward_solver.bottom_dofs
        ## thermal conductivity and heat source;
        self.h, self.q    = forward_solver.h, forward_solver.q
        ## temperature;
        self.T            = forward_solver.T
        ## PETSc options.
        self.petsc_opts = forward_solver.petsc_opts
        # Read observed temperature, noise level and ampltidue of regularization term.
        self.T_obs = fem.Function(self.V, name="ObservedTemperature")
        if isinstance(T_obs, fem.Function):
            assert T_obs.function_space == self.V
            self.T_obs.x.array[:] = T_obs.x.array[:]
        else:
            self.T_obs.x.array[:] = np.asarray(T_obs, dtype=float)
        self.T_obs.x.scatter_forward()
        self.sigma2 = sigma**2
        self.alpha = alpha

        # Define domain boundary conditions. Similar to those in the forward solver, except for DBC_value:
        ## 1) Dirichlet BC at the bottom.
        self.bcs = [fem.dirichletbc(
                        PETSc.ScalarType(DBC_value),
                        self.bottom_dofs,
                        self.V
                    )]
        ## 2) Neumann BC on the other three edges (insulated, zero flux).
        ## Note: No explicit Neumann BC is needed in the weak form.

        # Define the weak form of the adjoint equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        self.a = ufl.inner(self.h.function * ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = (1./self.sigma2) * ufl.inner((self.T - self.T_obs), v) * ufl.dx

        # Specify options for the PETSc KSP linear system solver.
        ## Inherit PETSc options from forward solver if none provided
        opts = petsc_opts if petsc_opts is not None else self.petsc_opts

        # Set up the linear variational problem, lhs=self.a, rhs=self.L, bcs=self.bcs.
        self.lambda_L = fem.Function(self.V, name="AdjointState")
        self.problem = fem.petsc.LinearProblem(
            self.a, self.L,
            u=self.lambda_L,
            bcs=self.bcs,
            petsc_options=opts
        )

    def solve(self) -> fem.Function:
        """
        Solve the adjoint equation on the domain, using the defined weak form and boundary conditions.

        Returns
        -------
        lambda_L : the Lagrangian multiplier function, aka adjoint state vector.
        """
        self.problem.solve()
        return self.lambda_L

    def assemble_gradient(self) -> PETSc.Vec:
        r"""
        Assemble the gradient dJ/dh.

        Returns
        -------
        array-like : the assembled gradient vector dJ/dh.
        """
        # Total derivative of the objective function J with respect to h:
        ## dJ/dh = (\lambda^T)\cdot g[T(h), h] + \partial J / partial h,
        ##       = -\nablaT\cdot\nabla\lambda + \alpha\lapl h.
        v = ufl.TestFunction(self.V)
        grad_expr = (
                - ufl.inner(ufl.grad(self.T), ufl.grad(self.lambda_L))*v \
                + self.alpha * ufl.inner(ufl.grad(self.h.function), ufl.grad(v))
            ) * ufl.dx
        grad_form = fem.form(grad_expr)
        grad_vec = fem.petsc.assemble_vector(grad_form)
        grad_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return grad_vec

    def update_gradient(self) -> None:
        """
        Update the gradient in self._grad.
        """
        self._grad = self.assemble_gradient()

    @property
    def grad(self) -> np.ndarray:
        if not hasattr(self, "_grad"):
            raise RuntimeError("Call update_gradient() first.")
        return self._grad
