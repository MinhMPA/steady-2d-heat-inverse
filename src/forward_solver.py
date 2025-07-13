from typing import Union, Callable
import numpy as np
from mpi4py import MPI
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
    Quick prototype of a forward solver for the steady-state Poisson heat equation on a 2D unit square.
    Dirichlet boundary condition is T = 300 K on the bottom edge (y == 0).
    Neumann boundary conditions are insulated (zero flux) on the other three edges.
    """
    def __init__(self,
                 nmesh: int = 64,
                 mesh_type: str = 'quadrilateral',
                 h: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
                 q: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
                 DBC_value: float = 300.0,
                 KSP_opts: dict = None):
        """
        Parameters
        ----------
        nmesh    : number of mesh nodes per side of the unit square, i.e. (x,y) in [0,1]. Default=64.
        h        : thermal conductivity, accept either
                     - float or fem.Constant, a fem.Constant on the domain;
                     - callable(x,y), a spatially-varying fem.Function.interpolate(callable) on the domain.
                    Default=1.
        q        : heat source, same allowed types as h. Default=1.
        DBC_value : Dirichlet BC at the bottom y=0. Default=300.
        KSP_opts : Dictionary of PETSc KSP options. Default=None.
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

        # Define domain boundary conditions.
            ## 1) Dirichlet BC at the bottom
        bottom = mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim-1,
            lambda x: np.isclose(x[1], 0.0)
        )
        dofs = fem.locate_dofs_topological(self.V,
                                           self.mesh.topology.dim-1,
                                           bottom)
        self.bcs = [fem.dirichletbc(
                        PETSc.ScalarType(DBC_value),
                        dofs,
                        self.V
                    )]
            ## 2) Neumann BC on the other three edges (insulated, zero flux)
            ## Note: No explicit Neumann BC is needed in the weak form.

        # Define the weak form of the Poisson equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        self.a = ufl.dot(self.h.function*ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = self.q.function * v * ufl.dx

        # Specify options for the PETSc KSP linear system solver.
        self.KSP_opts = KSP_opts or {
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'ksp_rtol': 1e-8
        }

    def solve(self):
        """
        Solve the steady-state heat equation on the domain, using the defined weak form and boundary conditions.
        Returns
        -------
        T : fem.Function, the temperature distribution on the mesh.
        """
        # Set up the linear variational problem, lhs=self.a, rhs=self.L, bcs=self.bcs, solver_options=self.KSP_opts.
        self.problem = LinearProblem(self.a,
                                     self.L,
                                     bcs=self.bcs,
                                     petsc_options=self.KSP_opts)
        T = self.problem.solve()
        T.name = "Temperature"
        self.T = T
        return T

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
            xdmf.write_function(wrapped_hfunc)
            xdmf.write_function(wrapped_qfunc)

    # Visualize output temperature.
    def plot_output_temperature(self, zero_point: float = 300.0, **kwargs):
        """
        Plot the temperature distribution on a pyvista.UnstructuredGrid.
        Parameters
        ----------
        cmap       : str, colormap. Default="viridis".
        zero_point : float, the "zero-point" temperature to be subtracted from the surface temperature T(x,y). Default=300., the bottom boundary temperature.
        **kwargs     : additional keyword arguments, see `plotting_utils.plot_scalar_mesh()` for details.
        """
        assert hasattr(self, 'T'), "No solution available. Call solve() first."
        vals = self.T.x.array[: self.mesh.geometry.x.shape[0]] - zero_point
        if zero_point != 0.0:
            print(r"Plotting relative temperature distribution DeltaT =T-T_0 with T_0=%.1fK." %zero_point)
            grid_plot = plot_scalar_mesh(self.mesh, vals, "ΔT [K]", cmap="viridis", **kwargs)
        else:
            print("Plotting absolute temperature distribution T(x,y).")
            grid_plot = plot_scalar_mesh(self.mesh, vals, "T [K]", cmap="viridis", **kwargs)
        return grid_plot
