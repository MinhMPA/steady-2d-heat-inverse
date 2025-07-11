from typing import Union, Callable
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import pyvista as pv

# dolphinx imports
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

class SteadyHeatForwardSolver2D:
    """
    Quick prototype of a forward solver for the steady-state Poisson heat equation on a 2D unit square.
    Dirichlet boundary condition is T = 300 K on the bottom edge (y == 0).
    Neumann boundary conditions are insulated (zero flux) on the other three edges.
    """
    def __init__(self,
                 nmesh: int = 64,
                 mesh_type: str = 'quadrilateral',
                 h: Union[float, fem.Constant, Callable] = 1.0,
                 q: Union[float, fem.Constant, Callable] = 1.0,
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
        # Set up the problem domain and function space on a unit square mesh.
        if mesh_type not in ['quadrilateral','triangle']:
            raise ValueError(f"Unsupported mesh type: {mesh_type}. Supported types: ['quadrilateral','triangle'].")
        elif mesh_type == 'quadrilateral':
            self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.quadrilateral)
        else:
            self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.triangle)
        x = ufl.SpatialCoordinate(self.mesh)
        self.V    = fem.functionspace(self.mesh, ('Lagrange', 1))

        #  Define thermal conductivity and heat source as domain coefficients.
        self.h = self._build_domain_func(h)
        self.q = self._build_domain_func(q)

        # Define domain boundary conditions.
            # 1) Dirichlet BC at the bottom
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
            # 2) Neumann BC on the other three edges (insulated, zero flux)
            # Note: No explicit Neumann BC is needed in the weak form.

        # Define the weak form of the Poisson equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        self.a = ufl.dot(self.h*ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = self.q * v * ufl.dx

        # Specify options for the PETSc KSP linear system solver.
        self.KSP_opts = KSP_opts or {
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'ksp_rtol': 1e-8
        }

    def _build_domain_func(self, param):
        """
        Return either a fem.Constant or interpolated Function on the domain.
        Parameters
        ----------
        param : float, fem.Constant or Callable
            - float, returns a fem.Constant on the mesh.
            - fem.Constant, simply returns the same fem.Constant.
            - Callable, returns the fem.Function.interpolate(callable).
        Returns
        -------
        f : fem.Constant or fem.Function, a domain function.
        """
        if isinstance(param, (int, float)):
            return fem.Constant(self.mesh,PETSc.ScalarType(param))
        if isinstance(param, fem.Constant):
            return param
        if callable(param):
            f = fem.Function(self.V)
            f.interpolate(param)
            return f
        raise TypeError(f"Cannot build coeff from {param!r}")

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

    def plot(self,
             cmap: str = 'viridis',
             zero_point: float = 300.0,
             show_edges: bool = False,
             ):
        """
        Plot the temperature distribution on a pyvista.UnstructuredGrid.
        Parameters
        ----------
        cmap       : str, colormap. Default="viridis".
        zero_point : float, the "zero-point" temperature to be subtracted from the surface temperature T(x,y). Default=300., i.e. the bottom boundary temperature.
        show_edges : bool, whether to plot mesh edges. Default=False.
        """
        assert hasattr(self, 'T'), "No solution available. Call solve() first."

        # Return early if not the root process.
        if MPI.COMM_WORLD.rank != 0:
            return

        # Retrieve the mesh topology, cell types, and geometry.
        topology, cell_types, geometry = vtk_mesh(self.mesh)

        # Setup and fill in the pv.UnstructuredGrid.
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["DeltaT"] = self.T.x.array[: geometry.shape[0]] - boundary_value

        # Plot the temperature distribution.
        grid.plot(
            scalars="DeltaT", cmap=cmap, show_edges=show_edges,
            scalar_bar_args=dict(
                fmt="%.2e", n_labels=5,
                title="T-T0", font_family="arial",
                title_font_size=20, label_font_size=14
            )
        )
