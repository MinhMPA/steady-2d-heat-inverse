"""Fixed-sensor conductivity posterior and its discrete-adjoint force."""

# type imports
from dataclasses import dataclass, replace
import hashlib

# numerical imports
import numpy as np

# pde imports
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
import ufl

# local imports
from conductivity_prior import BoundGaussianPrior, _ordered_cells
from heat_model import FrozenHeatModel, HeatNumerics, canonical_dof_indices
from temperature_observations import SensorOperator, TemperatureObservations


@dataclass(frozen=True)
class SolveCounts:
    r"""Actual attempted heat/adjoint solves and their cumulative iterations."""

    heat: int
    adjoint: int
    heat_iterations: int
    adjoint_iterations: int


@dataclass(frozen=True)
class PosteriorEvaluation:
    r"""A copied field, sensor likelihood, and Gaussian-coordinate potential force."""

    z: np.ndarray
    h: np.ndarray
    phi: float
    grad_phi: np.ndarray
    potential: float
    gradient: np.ndarray
    sensor_temperature: np.ndarray
    heat_iterations: int
    adjoint_iterations: int

    def __post_init__(self):
        for name in ("z", "h", "grad_phi", "gradient", "sensor_temperature"):
            object.__setattr__(
                self, name, np.array(getattr(self, name), dtype=np.float64, copy=True)
            )
        for name in ("phi", "potential"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("heat_iterations", "adjoint_iterations"):
            object.__setattr__(self, name, int(getattr(self, name)))


class HeatSolveFailure(RuntimeError):
    r"""An unconverged finite-element solve, fatal to this posterior calculation."""

    def __init__(self, stage, position, reason, residual_norm, iterations):
        self.stage = stage
        self.position = np.array(position, dtype=np.float64, copy=True)
        self.reason = int(reason)
        self.residual_norm = float(residual_norm)
        self.iterations = int(iterations)
        super().__init__(
            f"{stage} solve failed: PETSc reason={self.reason}, "
            f"residual_norm={self.residual_norm}, iterations={self.iterations}."
        )


class NonfiniteProposal(ValueError):
    r"""An unrepresentable proposal; the preceding physical workspace stays usable."""

    def __init__(self, stage, position):
        self.stage = stage
        self.position = np.array(position, dtype=np.float64, copy=True)
        super().__init__(f"Nonfinite or unrepresentable proposal at {stage}.")


class ConductivityPosterior:
    r"""Evaluate :math:`U(z)=\frac12\|\Sigma^{-1/2}(BT-d)\|^2+\frac12z^Tz`.

    Parameters
    ----------
    model : FrozenHeatModel
        Frozen mesh, source, and imposed temperatures; caller fields are untouched.
    observations : TemperatureObservations
        Fixed physical sensors with independent known Gaussian errors.
    prior : BoundGaussianPrior
        Already-bound full-field Gaussian log-conductivity prior on this mesh.
    numerics : HeatNumerics
        Fixed heat and adjoint tolerances, independent of evaluation history.
    prefix : str
        Nonempty PETSc prefix; heat and adjoint receive distinct suffixes.

    Notes
    -----
    The private heat unknown is the excess above a fixed boundary reference.
    Both the likelihood and adjoint use the centered residual, avoiding loss of
    force accuracy from an arbitrary absolute temperature origin. Public fields,
    sensor predictions, the frozen physical model and observations retain their
    original physical temperatures.
    """

    def __init__(self, model, observations, prior, numerics, *, prefix):
        self._closed = False
        self._forward = None
        self._adjoint_solver = None
        self._adjoint_rhs = None
        self._adjoint_solution = None
        self._heat_count = self._adjoint_count = 0
        self._heat_iterations = self._adjoint_iterations = 0
        if not isinstance(model, FrozenHeatModel):
            raise TypeError("model must be a FrozenHeatModel.")
        if not isinstance(observations, TemperatureObservations):
            raise TypeError("observations must be TemperatureObservations.")
        if not isinstance(prior, BoundGaussianPrior):
            raise TypeError("prior must be an already-bound BoundGaussianPrior.")
        if not isinstance(numerics, HeatNumerics):
            raise TypeError("numerics must be HeatNumerics.")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("A nonempty distinct PETSc prefix is required.")
        # Prior coordinates must describe this physical discretization, not just
        # an equal-sized vector on another mesh. Source/boundaries do not define
        # the conductivity prior's spatial binding.
        digest = hashlib.sha256(model.cell_type.encode("ascii"))
        for array in (model.coordinates, _ordered_cells(model.cells)):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        if (
            prior.dimension != len(model.coordinates)
            or prior.record()["mesh_fingerprint"] != digest.hexdigest()
        ):
            raise ValueError("The bound conductivity prior must match the frozen mesh.")
        self._model = model
        self._observations = observations
        self._prior = prior
        self._numerics = numerics
        # A fixed scalar reference leaves grad(T), the source and all
        # conductivity derivatives unchanged, including for nonuniform imposed
        # temperatures. It depends only on the canonical physical boundary.
        self._temperature_reference = (
            float(model.boundary_values[0]) if len(model.boundary_values) else 0.0
        )
        centered_model = replace(
            model, boundary_values=model.boundary_values - self._temperature_reference
        )
        self._centered_boundary_values = centered_model.boundary_values
        self._centered_observations = observations.values - self._temperature_reference
        self._centered_observations.setflags(write=False)
        try:
            self._forward = centered_model.build_forward(
                prefix=prefix + "heat_", numerics=numerics
            )
            forward = self._forward
            # PETSc 3.25.4 can leave its process-wide KSPSolve nesting level
            # incremented after a native nonconvergence error. Let this private
            # solve return its reason, then raise our fatal physical diagnostic.
            # The caller's original KSP error policy is never modified.
            forward.problem.solver.setErrorIfNotConverged(False)
            self._order = canonical_dof_indices(forward.V)
            self._sensors = SensorOperator(forward.V, observations.points)
            self._adjoint = fem.Function(forward.V, name="SensorAdjoint")
            self._adjoint_rhs = forward.problem.b.duplicate()
            self._adjoint_solution = forward.problem.x.duplicate()
            self._adjoint_solver = PETSc.KSP().create(forward.mesh.comm)
            solver = self._adjoint_solver
            solver.setOptionsPrefix(prefix + "adjoint_")
            solver.setType(numerics.ksp_type)
            solver.getPC().setType(numerics.pc_type)
            solver.setTolerances(
                rtol=numerics.rtol, atol=numerics.atol, max_it=numerics.max_it
            )
            solver.setErrorIfNotConverged(False)
            solver.setInitialGuessNonzero(False)
            v = ufl.TestFunction(forward.V)
            self._gradient_form = fem.form(
                -ufl.inner(ufl.grad(forward.T), ufl.grad(self._adjoint)) * v * ufl.dx
            )
        except Exception:
            self.close()
            raise

    @property
    def prior(self) -> BoundGaussianPrior:
        r"""The bound physical prior used by every evaluation."""
        return self._prior

    @property
    def observations(self) -> TemperatureObservations:
        r"""The fixed physical measurements and their noise law."""
        return self._observations

    @property
    def solve_counts(self) -> SolveCounts:
        r"""Return an immutable snapshot of actual solves, including failed attempts."""
        return SolveCounts(
            self._heat_count,
            self._adjoint_count,
            self._heat_iterations,
            self._adjoint_iterations,
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("The conductivity posterior is closed.")

    def _vector(self, value, name):
        value = np.asarray(value)
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real.")
        value = np.array(value, dtype=np.float64, copy=True)
        if value.shape != (self.prior.dimension,):
            raise ValueError(f"{name} must have shape ({self.prior.dimension},).")
        return value

    def _solve(self, stage, position):
        solver = (
            self._forward.problem.solver if stage == "heat" else self._adjoint_solver
        )
        if stage == "heat":
            self._heat_count += 1
        else:
            self._adjoint_count += 1
        error = None
        try:
            solver.setInitialGuessNonzero(False)
            if stage == "heat":
                self._forward.problem.x.set(0.0)
                self._forward.T.x.array[:] = 0.0
                self._forward.solve()
            else:
                self._adjoint_solution.set(0.0)
                solver.setOperators(self._forward.problem.A)
                solver.solve(self._adjoint_rhs, self._adjoint_solution)
                self._adjoint.x.array[:] = self._adjoint_solution.array_r
                self._adjoint.x.scatter_forward()
        except PETSc.Error as exc:
            error = exc
        iterations = int(solver.getIterationNumber())
        reason = int(solver.getConvergedReason())
        residual_norm = float(solver.getResidualNorm())
        if stage == "heat":
            self._heat_iterations += iterations
        else:
            self._adjoint_iterations += iterations
        if error is not None or reason <= 0:
            failure = HeatSolveFailure(
                stage, position, reason, residual_norm, iterations
            )
            self.close()
            raise failure from error
        # Dirichlet values are known constraints, not uncertain PDE unknowns.
        # The converged iterative solve can leave a finite algebraic residual
        # there. Restore only those values, preserving the free-node solution.
        constrained = self._order[self._model.boundary_dofs]
        if stage == "heat":
            self._forward.problem.x.array[constrained] = self._centered_boundary_values
            self._forward.T.x.array[constrained] = self._centered_boundary_values
            self._forward.T.x.scatter_forward()
        else:
            self._adjoint_solution.array[constrained] = 0.0
            self._adjoint.x.array[constrained] = 0.0
            self._adjoint.x.scatter_forward()
        return iterations

    def _predict(self, h, position):
        self._forward.h.function.x.array[self._order] = h
        self._forward.h.function.x.scatter_forward()
        iterations = self._solve("heat", position)
        excess = self._forward.T.x.array[self._order]
        if not np.isfinite(excess).all():
            raise NonfiniteProposal("temperature", position)
        centered_sensors = self._sensors.predict(self._forward.T.x.array)
        if not np.isfinite(centered_sensors).all():
            raise NonfiniteProposal("temperature", position)
        # Reconstruct physical output only after solving and interpolating the
        # excess. Never subtract these rounded public temperatures for the force.
        with np.errstate(over="ignore", invalid="ignore"):
            temperature = excess + self._temperature_reference
            sensors = centered_sensors + self._temperature_reference
        temperature[self._model.boundary_dofs] = self._model.boundary_values
        if not np.isfinite(temperature).all() or not np.isfinite(sensors).all():
            raise NonfiniteProposal("temperature", position)
        return temperature, sensors, centered_sensors, iterations

    def predict_h(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Predict canonical nodal temperatures and fixed physical sensor readings.

        Parameters
        ----------
        h : numpy.ndarray
            Positive finite physical conductivity in canonical node order.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Copied full temperature and sensor values; only one heat solve is counted.
        """
        self._require_open()
        h = self._vector(h, "Conductivity")
        if not np.isfinite(h).all() or np.any(h <= 0):
            raise NonfiniteProposal("conductivity", h)
        try:
            temperature, sensors, _, _ = self._predict(h, h)
            return temperature, sensors
        except NonfiniteProposal:
            raise
        except Exception:
            self.close()
            raise

    def evaluate(self, z: np.ndarray) -> PosteriorEvaluation:
        r"""Return sensor likelihood and total Gaussian-coordinate energy and force.

        Parameters
        ----------
        z : numpy.ndarray
            Full standard-normal coordinates in canonical physical node order.

        Returns
        -------
        PosteriorEvaluation
            Copied physical predictions, energies, derivatives, and solve iterations.
        """
        self._require_open()
        z = self._vector(z, "Position")
        if not np.isfinite(z).all():
            raise NonfiniteProposal("position", z)
        try:
            h = self.prior.to_h(z)
            if not np.isfinite(h).all() or np.any(h <= 0):
                raise NonfiniteProposal("conductivity", z)
            with np.errstate(over="ignore", invalid="ignore"):
                gaussian_energy = float(0.5 * (z @ z))
            if not np.isfinite(gaussian_energy):
                raise NonfiniteProposal("potential", z)
            _, sensors, centered_sensors, heat_iterations = self._predict(h, z)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = centered_sensors - self._centered_observations
                standardized = residual / self.observations.sigma
                phi = float(0.5 * (standardized @ standardized))
                weights = standardized / self.observations.sigma
                potential = phi + gaussian_energy
            if not np.isfinite(potential) or not np.isfinite(weights).all():
                raise NonfiniteProposal("likelihood", z)
            # B^T acts on a dual sensor vector directly. Constrained temperature
            # variations vanish, including for sensors on the imposed boundary.
            with np.errstate(over="ignore", invalid="ignore"):
                rhs = self._sensors.transpose(weights)
            rhs[self._forward.bottom_dofs] = 0.0
            # Duplicate sensors can overflow their accumulated dual weights
            # even when every weight and the likelihood energy are finite.
            if not np.isfinite(rhs).all():
                raise NonfiniteProposal("force", z)
            self._adjoint_rhs.array[:] = rhs
            adjoint_iterations = self._solve("adjoint", z)
            dual_h = assemble_vector(self._gradient_form)
            try:
                dual_h.ghostUpdate(
                    addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    grad_log_h = h * dual_h.array_r[self._order]
            finally:
                dual_h.destroy()
            if not np.isfinite(grad_log_h).all():
                raise NonfiniteProposal("force", z)
            grad_phi = self.prior.pullback(z, grad_log_h)
            gradient = grad_phi + z
            if not np.isfinite(gradient).all():
                raise NonfiniteProposal("force", z)
            return PosteriorEvaluation(
                z,
                h,
                phi,
                grad_phi,
                potential,
                gradient,
                sensors,
                heat_iterations,
                adjoint_iterations,
            )
        except NonfiniteProposal:
            raise
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        r"""Release owned heat/adjoint PETSc objects; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        for name in ("_adjoint_solver", "_adjoint_rhs", "_adjoint_solution"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)
        forward = getattr(self, "_forward", None)
        if forward is not None:
            problem = forward.problem
            # LinearProblem's own finalizer will inspect these same attributes.
            # Clear every destroyed reference so it cannot destroy a live alias
            # twice when Python later releases the private heat workspace.
            for name in ("_solver", "_A", "_b", "_x", "_P_mat"):
                obj = getattr(problem, name, None)
                if obj is not None:
                    obj.destroy()
                    setattr(problem, name, None)
            self._forward = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        self.close()
