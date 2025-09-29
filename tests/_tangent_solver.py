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


class _SteadyHeat2DTangentSolver(SteadyHeat2DForwardSolver):
    r"""
    Tangent solver for the tangent equation of the steady-state Poisson heat equation on a 2D unit square:
     \nabla(h\nabla\delta T) = -\nabla(\delta h\nabla T),
     where \delta T(\delta h) is the variation of the temperature field T, tangential to the variation \delta h in the thermal conductivity h.
    Dirichlet boundary condition on the bottom wall: \delta T(y=0)=0.
    Neumann boundary conditions on the other three insulated walls: \nabla\delta T\cdot\hat{n}=0.
    """

    def __init__(
        self,
        forward: SteadyHeat2DForwardSolver,
        petsc_opts: dict | None = None,
    ):
        """
        Parameters
        ----------
        forward         : forward solver instance, providing mesh, V, bottom_dofs, h, q, T.
        """
        # Copy forward solver attributes:
        ## mesh, function space, botom_dofs;
        self.mesh, self.V, self.bottom_dofs = (
            forward.mesh,
            forward.V,
            forward.bottom_dofs,
        )
        ## thermal conductivity and heat source;
        self.h, self.q = forward.h, forward.q
        ## temperature;
        self.T = forward.T
        ## PETSc options.
        self.petsc_opts = forward.petsc_opts

        # Define domain boundary conditions. Similar to those in the forward solver, except for DBC_value=0.0:
        ## 1) Dirichlet BC at the bottom.
        zero_DBC = fem.Constant(self.V.mesh, PETSc.ScalarType(0.0))
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(zero_DBC), self.bottom_dofs, self.V)
        ]
        ## 2) Neumann BC on the other three edges (insulated, zero flux).
        ## Note: No explicit Neumann BC is needed in the weak form.

        # Define the weak form of the forward tangent equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        ## Placeholder for delta_h
        self._delta_h = fem.Function(
            self.V, name="DeltaThermalConductivity_Placeholder"
        )
        self.a = ufl.inner(self.h.function * ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = -ufl.inner(self._delta_h * ufl.grad(self.T), ufl.grad(v)) * ufl.dx

        # Specify options for the PETSc KSP linear system solver.
        ## Inherit PETSc options from forward solver if none provided
        opts = petsc_opts if petsc_opts is not None else self.petsc_opts

        # Set up the linear variational problem, lhs=self.a, rhs=self.L, bcs=self.bcs.
        self.dT = fem.Function(self.V, name="TemperatureVariation")
        self.problem = LinearProblem(
            self.a, self.L, u=self.dT, bcs=self.bcs, petsc_options=opts
        )

    def solve(self):
        r"""
        Solve the tangent equation on the domain, using the defined weak form and boundary conditions.

        Returns
        -------
        dT : \delta T, the linear variation in temperature due to the variation in thermal conducitivity \delta h.
        """
        # Solve the tangent linear problem
        self.problem.solve()
        return self.dT

    def assemble_jvp(self, delta_h: fem.Function) -> fem.Function:
        r"""
        Assemble Jacobian-vector product:
            \delta T = (dT/dh)(\delta h),
        where (dT/dh) is the Jacobian of the conductivity-to-temperature map, and \delta h is the tangential vector of variation in thermal conductivity, along which the jvp is evaluated.

        Parameters
        ----------
        delta_h : \delta h, the vector of thermal conductivity variation.

        Returns
        -------
        dT : \delta T, the corresponding temperature variation.
        """
        # Check input
        assert isinstance(delta_h, fem.Function)
        assert delta_h.function_space == self.V
        # Update placeholder _delta_h with input delta_h
        self._delta_h.x.array[:] = delta_h.x.array[:]
        self._delta_h.x.scatter_forward()
        # Solve the tangent linear problem
        self.problem.solve()
        return self.dT
