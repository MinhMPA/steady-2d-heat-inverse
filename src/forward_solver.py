# type imports
from typing import Union, Callable

# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# pde imports
from petsc4py import PETSc
import ufl

# dolphinx imports
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh
from dolfinx.io import XDMFFile

# local imports
from domain_coefficient import ThermalConductivity, HeatSource
from plotting_utils import plot_scalar_mesh

class SteadyHeatForwardSolver2D:
    """
    Forward solver for the steady-state Poisson heat equation on a 2D unit square:
        \del\cdot(h\del T) = -q .
    Dirichlet boundary condition on the bottom wall: T(y=0)=300K.
    Neumann boundary conditions on the other three insulated walls: \del T\cdot\hat{n}=0.
    """
    def __init__(self,
                 nmesh: int = 64,
                 mesh_type: str = 'quadrilateral',
                 h: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
                 q: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
                 DBC_value: float = 300.0,
                 petsc_opts: dict = None):
        """
        Parameters
        ----------
        nmesh      : number of mesh nodes per side of the unit square, i.e. (x,y) in [0,1].
        h          : thermal conductivity, accept either
                     - float or fem.Constant, a fem.Constant on the domain;
                     - callable(x,y), a spatially-varying fem.Function.interpolate(callable) on the domain.
        q          : heat source, same allowed types as h.
        DBC_value  : Dirichlet BC for T at the bottom, i.e. T(y=0)=T0.
        petsc_opts : Dictionary of PETSc options.
        """
        # Define the problem domain, discretized on a unit square mesh. Two mesh types are supported: 'quadrilateral' and 'triangle'.
        if mesh_type not in ['quadrilateral','triangle']:
            raise ValueError(f"Unsupported mesh type: {mesh_type}. Supported types: ['quadrilateral','triangle'].")
        elif mesh_type == 'quadrilateral':
            self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.quadrilateral)
        else:
            self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.triangle)
        x = ufl.SpatialCoordinate(self.mesh)
        # Define the function space on the domain mesh, using Lagrange elements of degree 1.
        self.V    = fem.functionspace(self.mesh, ('Lagrange', 1))

        # Define thermal conductivity and heat source as domain coefficients.
        self.h = ThermalConductivity(h, self.mesh, self.V)
        self.q = HeatSource(q, self.mesh, self.V)

        # Define domain boundary conditions:
        ## 1) Dirichlet BC at the bottom.
        bottom = mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim-1,
            lambda x: np.isclose(x[1], 0.0)
        )
        self.bottom_dofs = fem.locate_dofs_topological(self.V,
                                           self.mesh.topology.dim-1,
                                           bottom)
        self.bcs = [fem.dirichletbc(
                        PETSc.ScalarType(DBC_value),
                        self.bottom_dofs,
                        self.V
                    )]
        ## 2) Neumann BC on the other three edges (insulated, zero flux).
        ## Note: No explicit Neumann BC is needed in the weak form.

        # Define the weak form of the Poisson equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        ## ufl.inner = ufl.dot in this case
        self.a = ufl.inner(self.h.function*ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = self.q.function * v * ufl.dx

        # Specify options for the PETSc KSP linear system solver.
        ## Default options are set to use the conjugate gradient method with hypre preconditioner.
        default_opts = {
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'ksp_rtol': 1e-8
        }
        self.petsc_opts = default_opts | (petsc_opts or {})

        # Set up the linear variational problem, lhs=self.a, rhs=self.L, bcs=self.bcs.
        self.T = fem.Function(self.V, name="Temperature")
        self.problem = LinearProblem(self.a,
                                     self.L,
                                     u=self.T,
                                     bcs=self.bcs,
                                     petsc_options=self.petsc_opts)

    # Main driver method to solve the steady-state heat equation as a linear variational problem.
    def solve(self) -> fem.Function:
        """
        Solve the steady-state heat equation on the domain, using the defined weak form and boundary conditions.
        Returns
        -------
        T : the temperature distribution on the mesh.
        """
        self.problem.solve()
        return self.T

    # Supporting method to inject a Gaussian noise field into the solution.
    def add_noise(self, mu: float = 0.0, sigma: float = 1.0, seed: int | None = None) -> fem.Function:
        """
        Add uncorrelated Gaussian noise N(mu, sigma^2) to the solution T(x,y).
        Default noise distribution is a standard normal N(0,1).

        Parameters
        ----------
        mu : mean of the noise to add (in the same unit as T).
        sigma : standard deviation of the noise to add (in the same unit as T).
        seed : random number generator seed, for reproducibility.
        """
        if not hasattr(self, "T"):
            raise RuntimeError("No solution available. Call solve() first.")

        self.T_obs = fem.Function(self.T.function_space, name="ObservedTemperature")
        # Number of DOFs local to the MPI rank
        n_local_dofs = self.mesh.geometry.x.shape[0]
        self.T_obs.x.array[:n_local_dofs] = self.T.x.array[:n_local_dofs]

        # Generate noise on rank 0 and broadcast noise to all ranks
        if MPI.COMM_WORLD.rank == 0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=mu, scale=sigma, size=n_local_dofs)
        else:
            noise = np.empty(n_local_dofs, dtype=float)
        MPI.COMM_WORLD.Bcast(noise, root=0)

        # Inject noise to T_obs and broadcast T_obs to all ranks
        self.T_obs.x.array[:n_local_dofs] += noise
        self.T_obs.x.scatter_forward()
        return self.T_obs

    # Write to XDMF file.
    def export_xdmf(self, filename: str):
        """
        Write the domain mesh, output temperature, input thermal conductivity and input heat source to disk in XDMF/HDF5 format.

        Parameters
        ----------
        filename : str, the output filename for the XDMF file.
        """
        def _wrap_constant_in_function(coefficient):
            """
            Wrap a fem.Constant in a fem.Function for export.
            Only meant to be used for ThermalConductivity and HeatSource, which are either fem.Function or fem.Constant.
            """
            if isinstance(coefficient, fem.Function):
                return coefficient
            else:
                f = fem.Function(self.V)
                f.interpolate(lambda x: np.full(x.shape[1], coefficient.value, dtype=default_scalar_type))
                return f
        wrapped_hfunc = _wrap_constant_in_function(self.h.function)
        wrapped_hfunc.name = "ThermalConductivity"
        wrapped_qfunc = _wrap_constant_in_function(self.q.function)
        wrapped_qfunc.name = "HeatSource"
        with XDMFFile(self.mesh.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.T)
            xdmf.write_function(self.T_obs)
            xdmf.write_function(wrapped_hfunc)
            xdmf.write_function(wrapped_qfunc)

    # Visualize output temperature.
    def plot_output_temperature(self, zero_point: float = 300.0, noiseless: bool = True, **kwargs):
        """
        Plot the temperature distribution on a pyvista.UnstructuredGrid.

        Parameters
        ----------
        zero_point : the "zero-point" temperature to be subtracted from the surface temperature T(x,y).
        noiseless  : whether to plot the noiseless temperature `T` or the noise-injected temperature `T_obs`.
        **kwargs   : additional keyword arguments, see `plotting_utils.plot_scalar_mesh()` for details.
        """
        if noiseless:
            assert hasattr(self, 'T'), "No solution available. Call solve() first."
            vals = self.T.x.array[: self.mesh.geometry.x.shape[0]] - zero_point
        else:
            assert hasattr(self, 'T_obs'), "No solution available. Call solve() first."
            vals = self.T_obs.x.array[: self.mesh.geometry.x.shape[0]] - zero_point
        if zero_point != 0.0:
            print(r"Plotting relative temperature distribution DeltaT =T-T_0 with T_0=%.1fK." %zero_point)
            grid_plot = plot_scalar_mesh(self.mesh, vals, "ΔT [K]", cmap="viridis", **kwargs)
        else:
            print("Plotting absolute temperature distribution T(x,y).")
            grid_plot = plot_scalar_mesh(self.mesh, vals, "T [K]", cmap="viridis", **kwargs)
        return grid_plot
