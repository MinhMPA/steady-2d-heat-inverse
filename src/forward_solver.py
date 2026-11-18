# type imports
from typing import Union, Callable
from itertools import count

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

_APPLICATION_PREFIXES = count()


def _linear_problem(*args, petsc_options, **kwargs):
    """Apply temporary solver options without leaving process-global entries.

    The qualified PETSc release does not apply prefixPush to option deletion.
    Use full keys for insertion and restoration, including failed setup.
    """
    problem = LinearProblem(*args, petsc_options=None, **kwargs)
    options = PETSc.Options()
    before = options.getAll()
    prefix = problem.solver.getOptionsPrefix()
    keys = [prefix + key for key in petsc_options]
    try:
        for key, value in petsc_options.items():
            options[prefix + key] = value
        problem.solver.setFromOptions()
    except Exception:
        _release_application_problem(problem)
        raise
    finally:
        for key in keys:
            if key in before:
                options[key] = before[key]
            elif options.hasName(key):
                del options[key]
    return problem


def _release_application_problem(problem):
    """Release an owned trial problem before its eventual finalizer."""
    if problem is not None:
        for name in ("_solver", "_A", "_b", "_x", "_P_mat"):
            obj = getattr(problem, name, None)
            if obj is not None:
                obj.destroy()
                setattr(problem, name, None)


class SteadyHeat2DForwardSolver:
    r"""
    Forward solver for the steady-state Poisson heat equation on a 2D unit square:
        \nabla\cdot(h\nabla T) = -q .

    Dirichlet boundary condition on the bottom wall: T(y=0)=300K.
    Neumann boundary conditions on the other three insulated walls: \nabla T\cdot\hat{n}=0.
    """

    def __init__(
        self,
        nmesh: int = 64,
        mesh_type: str = "quadrilateral",
        h: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
        q: Union[float, fem.Constant, fem.Expression, Callable] = 1.0,
        DBC_value: float = 300.0,
        petsc_opts: dict = None,
        tab_interpolator: str = "rbf",
    ):
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
        tab_interpolator : interpolation scheme for tabulated h/q input, "rbf" or "ct".
        """
        # Define the problem domain, discretized on a unit square mesh. Two mesh types are supported: 'quadrilateral' and 'triangle'.
        if mesh_type not in ["quadrilateral", "triangle"]:
            raise ValueError(
                f"Unsupported mesh type: {mesh_type}. Supported types: ['quadrilateral','triangle']."
            )
        elif mesh_type == "quadrilateral":
            self.mesh = mesh.create_unit_square(
                MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.quadrilateral
            )
        else:
            self.mesh = mesh.create_unit_square(
                MPI.COMM_WORLD, nmesh, nmesh, mesh.CellType.triangle
            )
        x = ufl.SpatialCoordinate(self.mesh)
        # Define the function space on the domain mesh, using Lagrange elements of degree 1.
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))

        # Define thermal conductivity and heat source as domain coefficients.
        self.h = ThermalConductivity(
            h, self.mesh, self.V, tab_interpolator=tab_interpolator
        )
        self.q = HeatSource(q, self.mesh, self.V, tab_interpolator=tab_interpolator)

        # Define domain boundary conditions:
        ## 1) Dirichlet BC at the bottom.
        bottom = mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0)
        )
        self.bottom_dofs = fem.locate_dofs_topological(
            self.V, self.mesh.topology.dim - 1, bottom
        )
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(DBC_value), self.bottom_dofs, self.V)
        ]
        ## 2) Neumann BC on the other three edges (insulated, zero flux).
        ## Note: No explicit Neumann BC is needed in the weak form.

        # Specify options for the PETSc KSP linear system solver.
        ## Default options are set to use the conjugate gradient method with hypre preconditioner.
        ## `ksp_error_if_not_converged` makes a failed solve raise instead of silently
        ## returning an unconverged state into the objective and gradient.
        default_opts = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_error_if_not_converged": True,
        }
        self.petsc_opts = default_opts | (petsc_opts or {})

        self._coefficient_generation = 0
        self._initialize_linear_problem("s2dhi_fwd_")

    @classmethod
    def from_discrete_data(
        cls,
        *,
        domain,
        h_values,
        q_values,
        boundary_dofs,
        boundary_values,
        petsc_opts,
        prefix,
    ):
        r"""Build a private heat workspace from native nodal physical data.

        Parameters
        ----------
        domain : dolfinx.mesh.Mesh
            Single-rank physical mesh.
        h_values, q_values : numpy.ndarray
            Native scalar nodal conductivity and source values.
        boundary_dofs, boundary_values : numpy.ndarray
            Native constrained indices and their imposed temperatures.
        petsc_opts : dict
            Fixed linear-solve settings.
        prefix : str
            Distinct PETSc options prefix.

        Returns
        -------
        SteadyHeat2DForwardSolver
            Independent heat fields and variational problem.
        """
        if domain.comm.size != 1:
            raise ValueError("A private heat workspace requires a single MPI rank.")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("A nonempty distinct PETSc prefix is required.")
        self = cls.__new__(cls)
        self.mesh = domain
        self.V = fem.functionspace(domain, ("Lagrange", 1))
        count = self.V.dofmap.index_map.size_local
        arrays = {}
        for name, values, size in (
            ("conductivity", h_values, count),
            ("source", q_values, count),
            ("boundary temperatures", boundary_values, len(boundary_dofs)),
        ):
            values = np.asarray(values)
            if np.iscomplexobj(values) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite real values.")
            if values.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},).")
            arrays[name] = values
        if np.any(arrays["conductivity"] <= 0.0):
            raise ValueError("Conductivity must be positive at every node.")
        dofs = np.asarray(boundary_dofs)
        if (
            dofs.ndim != 1
            or not np.issubdtype(dofs.dtype, np.integer)
            or np.any(dofs < 0)
            or np.any(dofs >= count)
            or len(np.unique(dofs)) != len(dofs)
        ):
            raise ValueError("Boundary DOFs must be unique valid integer indices.")
        h = fem.Function(self.V)
        q = fem.Function(self.V)
        h.x.array[:] = arrays["conductivity"]
        q.x.array[:] = arrays["source"]
        h.x.scatter_forward()
        q.x.scatter_forward()
        self.h = ThermalConductivity(h, domain, self.V)
        self.q = HeatSource(q, domain, self.V)
        boundary = fem.Function(self.V)
        boundary.x.array[dofs] = arrays["boundary temperatures"]
        boundary.x.scatter_forward()
        self.bottom_dofs = np.sort(dofs).astype(np.int32)
        self.bcs = [fem.dirichletbc(boundary, self.bottom_dofs)]
        self.petsc_opts = dict(petsc_opts) | {
            "ksp_error_if_not_converged": True,
            "ksp_initial_guess_nonzero": False,
        }
        self._coefficient_generation = 0
        self._initialize_linear_problem(prefix)
        return self

    def _initialize_linear_problem(self, prefix: str) -> None:
        r"""Build the common heat forms, preserving a matching temperature field."""
        # Define the weak form of the Poisson equation.
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        ## ufl.inner = ufl.dot in this case
        self.a = ufl.inner(self.h.function * ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.L = self.q.function * v * ufl.dx

        if not hasattr(self, "T") or self.T.function_space != self.V:
            self.T = fem.Function(self.V, name="Temperature")
        self.problem = _linear_problem(
            self.a,
            self.L,
            u=self.T,
            bcs=self.bcs,
            petsc_options_prefix=prefix,
            petsc_options=self.petsc_opts,
        )
        self.total_ksp_iterations = 0
        self.total_linear_solves = 0

    @property
    def coefficient_generation(self) -> int:
        """Count coefficient-handle replacements by explicit field application.

        Changing nodal values preserves the handle and this read-only count.
        Promoting a scalar coefficient changes it, requiring legacy adjoint and
        reconstruction objects to be reconstructed with the new heat forms.
        """
        return self._coefficient_generation

    def _solve_application(self):
        """Check application convergence explicitly, preserving legacy policy."""
        solver = self.problem.solver
        policy = solver.getErrorIfNotConverged()
        try:
            # Native PETSc exceptions can affect subsequent unrelated solves.
            # Only this reversible application uses an explicit reason check;
            # ordinary solve() retains the caller's original native policy.
            solver.setErrorIfNotConverged(False)
            self.solve()
            reason = int(solver.getConvergedReason())
            if reason <= 0:
                raise RuntimeError(
                    f"Heat solve did not converge during conductivity application (reason {reason})."
                )
            if not np.isfinite(self.T.x.array).all():
                raise RuntimeError(
                    "Heat solve returned nonfinite application temperatures."
                )
        finally:
            solver.setErrorIfNotConverged(policy)

    def apply_conductivity_field(
        self, values: np.ndarray, *, coordinates: np.ndarray
    ) -> fem.Function:
        """Apply a positive nodal conductivity and solve its temperature reversibly.

        Parameters
        ----------
        values : numpy.ndarray
            Positive finite conductivity at every node, labelled by coordinates.
        coordinates : numpy.ndarray
            Physical (N,2) labels matching this mesh to floating-point roundoff.
            No interpolation or transfer between meshes is performed.

        Returns
        -------
        dolfinx.fem.Function
            The existing temperature object containing the converged solution.
            A nodal conductivity retains its coefficient handle. A scalar is
            promoted only after a separate successful trial heat solve. Failed
            solves or adoption restore original handles, forms and field arrays.
        """
        from heat_model import (
            FrozenHeatModel,
            NodalField,
            coerce_field,
            canonical_dof_indices,
        )

        model = FrozenHeatModel.from_forward(self)
        physical = coerce_field(NodalField(values, coordinates), model)
        if np.any(physical <= 0):
            raise ValueError("Conductivity must be positive at every physical node.")
        order = canonical_dof_indices(self.V)
        old_temperature = self.T.x.array.copy()
        if isinstance(self.h.function, fem.Function):
            old_conductivity = self.h.function.x.array.copy()
            old_solution = self.problem.x.array.copy()
            try:
                self.h.function.x.array[order] = physical
                self.h.function.x.scatter_forward()
                self._solve_application()
                return self.T
            except BaseException:
                self.h.function.x.array[:] = old_conductivity
                self.h.function.x.scatter_forward()
                self.T.x.array[:] = old_temperature
                self.T.x.scatter_forward()
                self.problem.x.array[:] = old_solution
                self.problem.x.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                raise

        # A scalar coefficient has no independent nodal unknowns. Prepare a
        # complete trial problem so failed promotion cannot touch caller state.
        trial = SteadyHeat2DForwardSolver.__new__(SteadyHeat2DForwardSolver)
        trial.mesh, trial.V = self.mesh, self.V
        conductivity = fem.Function(self.V, name="ThermalConductivity")
        conductivity.x.array[order] = physical
        conductivity.x.scatter_forward()
        trial.h = ThermalConductivity(conductivity, self.mesh, self.V)
        trial.q, trial.bcs, trial.bottom_dofs = self.q, self.bcs, self.bottom_dofs
        trial.petsc_opts = dict(self.petsc_opts)
        trial._coefficient_generation = self.coefficient_generation
        trial_problem = final_problem = None
        old_h_attributes = self.h.__dict__.copy()
        old_forms = self.a, self.L, self.problem, self.coefficient_generation
        try:
            prefix = f"s2dhi_apply_trial_{next(_APPLICATION_PREFIXES)}_"
            trial._initialize_linear_problem(prefix)
            trial_problem = trial.problem
            # Preserve any tolerance/policy changes made directly on the
            # destination KSP as well as settings made through public setters.
            rtol, atol, divtol, max_it = self.get_ksp_tolerances()
            trial.problem.solver.setTolerances(
                rtol=rtol, atol=atol, divtol=divtol, max_it=max_it
            )
            trial.problem.solver.setErrorIfNotConverged(
                self.problem.solver.getErrorIfNotConverged()
            )
            trial._solve_application()
            # LinearProblem.u is read-only. Build the adopted forms through its
            # public constructor, preserving the original temperature object.
            final_problem = _linear_problem(
                trial.a,
                trial.L,
                u=self.T,
                bcs=self.bcs,
                petsc_options_prefix=f"s2dhi_apply_{next(_APPLICATION_PREFIXES)}_",
                petsc_options=self.petsc_opts,
            )
            final_problem.solver.setTolerances(
                rtol=rtol, atol=atol, divtol=divtol, max_it=max_it
            )
            final_problem.solver.setErrorIfNotConverged(
                self.problem.solver.getErrorIfNotConverged()
            )
            final_problem.solver.setInitialGuessNonzero(
                self.problem.solver.getInitialGuessNonzero()
            )
            final_problem.x.array[:] = trial_problem.x.array
            final_problem.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            self.h.__dict__.update(trial.h.__dict__)
            self.a, self.L, self.problem = trial.a, trial.L, final_problem
            self.T.x.array[:] = trial.T.x.array
            self.T.x.scatter_forward()
            self._coefficient_generation = old_forms[3] + 1
            return self.T
        except BaseException:
            self.h.__dict__.clear()
            self.h.__dict__.update(old_h_attributes)
            self.a, self.L, self.problem, self._coefficient_generation = old_forms
            self.T.x.array[:] = old_temperature
            self.T.x.scatter_forward()
            _release_application_problem(final_problem)
            raise
        finally:
            _release_application_problem(trial_problem)

    def get_ksp_tolerances(self) -> tuple[float, float, float, int]:
        """
        Return the current KSP tolerances from the wrapped PETSc solver.
        """
        return self.problem.solver.getTolerances()

    def get_ksp_rtol(self) -> float:
        """
        Return the current KSP relative tolerance.
        """
        rtol, _, _, _ = self.get_ksp_tolerances()
        return float(rtol)

    def set_ksp_tolerances(
        self,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        divtol: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """
        Update the PETSc KSP tolerances for the wrapped linear solve.
        """
        self.problem.solver.setTolerances(
            rtol=rtol, atol=atol, divtol=divtol, max_it=max_it
        )
        if rtol is not None:
            self.petsc_opts["ksp_rtol"] = rtol
        if atol is not None:
            self.petsc_opts["ksp_atol"] = atol
        if divtol is not None:
            self.petsc_opts["ksp_divtol"] = divtol
        if max_it is not None:
            self.petsc_opts["ksp_max_it"] = max_it

    def set_ksp_rtol(self, rtol: float) -> None:
        """
        Update only the PETSc KSP relative tolerance.
        """
        self.set_ksp_tolerances(rtol=rtol)

    # Main driver method to solve the steady-state heat equation as a linear variational problem.
    def solve(self) -> fem.Function:
        """
        Solve the steady-state heat equation on the domain, using the defined weak form and boundary conditions.
        Returns
        -------
        T : the temperature distribution on the mesh.
        """
        self.total_linear_solves += 1
        self.problem.solver.setIterationNumber(0)
        try:
            self.problem.solve()
        finally:
            self.total_ksp_iterations += int(self.problem.solver.getIterationNumber())
        return self.T

    # Supporting method to inject a Gaussian noise field into the solution.
    def add_noise(
        self, mu: float = 0.0, sigma: float = 1.0, seed: int | None = None
    ) -> fem.Function:
        r"""
        Add uncorrelated Gaussian noise N(\mu, \sigma^2) to the solution T(x,y).
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
                f.interpolate(
                    lambda x: np.full(
                        x.shape[1], coefficient.value, dtype=default_scalar_type
                    )
                )
                return f

        wrapped_hfunc = _wrap_constant_in_function(self.h.function)
        wrapped_hfunc.name = "ThermalConductivity"
        wrapped_qfunc = _wrap_constant_in_function(self.q.function)
        wrapped_qfunc.name = "HeatSource"
        with XDMFFile(self.mesh.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.T)
            if hasattr(self, "T_obs"):
                xdmf.write_function(self.T_obs)
            xdmf.write_function(wrapped_hfunc)
            xdmf.write_function(wrapped_qfunc)

    # Visualize output temperature.
    def plot_output_temperature(
        self, zero_point: float = 300.0, noiseless: bool = True, **kwargs
    ):
        """
        Plot the temperature distribution on a pyvista.UnstructuredGrid.

        Parameters
        ----------
        zero_point : the "zero-point" temperature to be subtracted from the surface temperature T(x,y).
        noiseless  : whether to plot the noiseless temperature `T` or the noise-injected temperature `T_obs`.
        **kwargs   : additional keyword arguments, see `plotting_utils.plot_scalar_mesh()` for details.
        """
        if noiseless:
            if not hasattr(self, "T"):
                raise RuntimeError("No solution available. Call solve() first.")
            vals = self.T.x.array[: self.mesh.geometry.x.shape[0]] - zero_point
        else:
            if not hasattr(self, "T_obs"):
                raise RuntimeError(
                    "No noisy observation available. Call add_noise() first."
                )
            vals = self.T_obs.x.array[: self.mesh.geometry.x.shape[0]] - zero_point
        if zero_point != 0.0:
            print(
                r"Plotting relative temperature distribution DeltaT =T-T_0 with T_0=%.1fK."
                % zero_point
            )
            grid_plot = plot_scalar_mesh(
                self.mesh, vals, "ΔT [K]", cmap="viridis", **kwargs
            )
        else:
            print("Plotting absolute temperature distribution T(x,y).")
            grid_plot = plot_scalar_mesh(
                self.mesh, vals, "T [K]", cmap="viridis", **kwargs
            )
        return grid_plot
