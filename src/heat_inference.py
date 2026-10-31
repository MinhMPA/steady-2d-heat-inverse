"""Full-field posterior optimization and complete inferred conductivity states."""

# type imports
from dataclasses import dataclass
import hashlib
import json
from itertools import count

# numerical imports
import numpy as np
from scipy.optimize import minimize

# local imports
from conductivity_posterior import ConductivityPosterior, NonfiniteProposal
from conductivity_prior import (
    BoundGaussianPrior,
    GaussianLogConductivity,
    _ordered_cells,
)
from heat_model import (
    FieldInput,
    FrozenHeatModel,
    HeatNumerics,
    NodalField,
    _real_values,
    coerce_field,
)
from temperature_observations import TemperatureObservations

_PREFIXES = count()


def _digest(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InferredState:
    r"""Every inferred coordinate, labelled by the fixed physical experiment.

    Parameters
    ----------
    coordinates : dict[str, numpy.ndarray]
        Complete declared coordinate arrays; this Gaussian field uses ``z``.
    problem_id : str
        Identity of the physical model, observations, noise law and prior.
    """

    coordinates: dict[str, np.ndarray]
    problem_id: str

    def __post_init__(self):
        # Retain every declared field; validation against a particular prior is
        # the inference problem's responsibility, rather than this container's.
        object.__setattr__(
            self,
            "coordinates",
            {
                name: np.array(value, copy=True)
                for name, value in self.coordinates.items()
            },
        )


@dataclass(frozen=True)
class PosteriorEstimate:
    r"""A log-conductivity mode candidate, with convergence status kept explicit.

    Parameters
    ----------
    state : InferredState
        Complete inferred coordinates belonging to the same posterior.
    conductivity : numpy.ndarray
        Positive physical field in canonical physical node order.
    objective : float
        Declared posterior potential at the returned state.
    success : bool
        Whether the requested optimization convergence criterion was achieved.
    message : str
        Optimizer termination reason, including any nonfinite trial.
    mode_coordinates : str
        Density coordinates defining this mode, ``log_conductivity`` here.
    """

    state: InferredState
    conductivity: np.ndarray
    objective: float
    success: bool
    message: str
    mode_coordinates: str = "log_conductivity"

    def __post_init__(self):
        object.__setattr__(
            self, "state", InferredState(self.state.coordinates, self.state.problem_id)
        )
        object.__setattr__(
            self,
            "conductivity",
            np.array(self.conductivity, dtype=np.float64, copy=True),
        )
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "message", str(self.message))


class HeatInverseProblem:
    r"""One frozen heat experiment and its full-field conductivity posterior.

    Parameters
    ----------
    forward : SteadyHeat2DForwardSolver
        Caller-owned physical heat model. Its current conductivity and
        temperature are preserved and excluded from the experiment identity.
    observations : TemperatureObservations
        Fixed physical sensor measurements and known independent errors.
    prior : GaussianLogConductivity or BoundGaussianPrior
        Full-field Gaussian log-conductivity specification or matching bound
        prior; physical parameters and evaluated mean are copied at construction.
    numerics : HeatNumerics, optional
        Fixed heat and adjoint tolerances; separately identified from physics.
    """

    def __init__(
        self, *, forward, observations, prior, numerics: HeatNumerics | None = None
    ):
        self._closed = False
        self._workspaces = []
        self._forward = forward
        try:
            self._model = FrozenHeatModel.from_forward(forward)
            if not isinstance(observations, TemperatureObservations):
                raise TypeError("observations must be TemperatureObservations.")
            self._observations = TemperatureObservations(
                observations.points, observations.values, observations.sigma
            )
            if numerics is None:
                numerics = HeatNumerics()
            if not isinstance(numerics, HeatNumerics):
                raise TypeError("numerics must be HeatNumerics.")
            # Own normalized scalar values, even if a future numerics container
            # permits mutable input scalars.
            self._numerics = HeatNumerics(
                rtol=numerics.rtol,
                atol=numerics.atol,
                max_it=int(numerics.max_it),
                ksp_type=numerics.ksp_type,
                pc_type=numerics.pc_type,
            )
            if isinstance(prior, BoundGaussianPrior):
                record = prior.record()
                mesh_digest = hashlib.sha256(self._model.cell_type.encode("ascii"))
                for array in (
                    self._model.coordinates,
                    _ordered_cells(self._model.cells),
                ):
                    mesh_digest.update(str(array.shape).encode("ascii"))
                    mesh_digest.update(array.tobytes())
                if record["mesh_fingerprint"] != mesh_digest.hexdigest():
                    raise ValueError(
                        "The bound conductivity prior must match the frozen mesh."
                    )
                prior = GaussianLogConductivity(
                    mu=NodalField(np.array(record["mu"]), self._model.coordinates),
                    kappa=record["kappa"],
                    tau=record["tau"],
                    h_ref=record["h_ref"],
                )
            if not isinstance(prior, GaussianLogConductivity):
                raise TypeError(
                    "prior must be GaussianLogConductivity or BoundGaussianPrior."
                )
            self._prior = prior.bind(self._model, forward.V)
            sigma = np.broadcast_to(
                self._observations.sigma, self._observations.values.shape
            )
            self._problem_id = _digest(
                {
                    "physical_model": self._model.fingerprint,
                    "sensor_points": self._observations.points.tolist(),
                    "sensor_values": self._observations.values.tolist(),
                    "sensor_sigma": sigma.tolist(),
                    "prior": self._prior.record(),
                    "coordinate_schema": self._prior.coordinate_schema,
                }
            )
            self._numerics_id = _digest(
                {
                    "rtol": self._numerics.rtol,
                    "atol": self._numerics.atol,
                    "max_it": self._numerics.max_it,
                    "ksp_type": self._numerics.ksp_type,
                    "pc_type": self._numerics.pc_type,
                }
            )
            self._posterior = self._new_posterior(
                prefix=f"heat_inference_{next(_PREFIXES)}_"
            )
        except Exception:
            self.close()
            raise

    @property
    def forward(self):
        r"""The supplied caller heat model, or ``None`` for a loaded problem."""
        return self._forward

    @property
    def posterior(self) -> ConductivityPosterior:
        r"""The declared conductivity posterior with its owned heat workspace."""
        return self._posterior

    @property
    def problem_id(self) -> str:
        r"""Immutable physical experiment and probability-law identity."""
        return self._problem_id

    @property
    def numerics_id(self) -> str:
        r"""Fixed heat/adjoint accuracy identity, separate from the physics."""
        return self._numerics_id

    def _require_open(self):
        if self._closed:
            raise RuntimeError("The heat inference problem is closed.")

    def _new_posterior(self, *, prefix: str) -> ConductivityPosterior:
        r"""Construct an independently owned workspace from frozen inputs."""
        self._require_open()
        posterior = ConductivityPosterior(
            self._model, self._observations, self._prior, self._numerics, prefix=prefix
        )
        self._workspaces.append(posterior)
        return posterior

    def initial_state(
        self, value: PosteriorEstimate | InferredState | FieldInput
    ) -> InferredState:
        r"""Carry a complete state or infer its coordinates from a physical field.

        Parameters
        ----------
        value : PosteriorEstimate, InferredState or FieldInput
            A complete state from this experiment, or a physical conductivity
            whose prior has an inverse initialization map.

        Returns
        -------
        InferredState
            Independent copies of every declared coordinate. Existing complete
            states do not require any inverse from physical conductivity.
        """
        self._require_open()
        if isinstance(value, PosteriorEstimate):
            value = value.state
        if isinstance(value, InferredState):
            if value.problem_id != self.problem_id:
                raise ValueError(
                    "Initial state belongs to a different inference problem/experiment."
                )
            coordinates = value.coordinates
        else:
            inverse = getattr(self._prior, "from_h", None)
            if not callable(inverse):
                raise ValueError(
                    "Physical-only initialization requires a prior inverse for the complete inferred state."
                )
            h = coerce_field(value, self._model)
            if np.any(h <= 0):
                raise ValueError("Initial conductivity must be positive.")
            recovered = inverse(h)
            coordinates = recovered if isinstance(recovered, dict) else {"z": recovered}
        schema = self._prior.coordinate_schema
        if set(coordinates) != set(schema):
            raise ValueError(
                "Initial state must contain the complete declared coordinate schema."
            )
        copied = {}
        for name, shape in schema.items():
            array = _real_values(coordinates[name], f"Initial coordinate {name}")
            if array.shape != tuple(shape):
                raise ValueError(
                    f"Initial coordinate {name} must have shape {tuple(shape)}."
                )
            copied[name] = array.copy()
        return InferredState(copied, self.problem_id)

    def optimize(
        self, *, initial_h: FieldInput, maxiter=500, gtol=1e-6, ftol=1e-12
    ) -> PosteriorEstimate:
        r"""Minimize the declared posterior potential over all field coordinates.

        The affine map from ``z`` to log conductivity has constant determinant,
        so this gives the same mode in log conductivity. No conductivity-density
        Jacobian or positivity bounds are added.

        Parameters
        ----------
        initial_h : FieldInput
            Positive physical conductivity on this frozen mesh.
        maxiter : int
            Positive maximum number of L-BFGS-B iterations.
        gtol, ftol : float
            Finite nonnegative gradient and relative-energy stopping tolerances.

        Returns
        -------
        PosteriorEstimate
            Finite mode candidate with the actual convergence status. A
            nonfinite trial returns the last finite state with ``success=False``;
            an unconverged heat/adjoint solve raises ``HeatSolveFailure``.
        """
        self._require_open()
        if (
            isinstance(maxiter, (bool, np.bool_))
            or not isinstance(maxiter, (int, np.integer))
            or maxiter <= 0
        ):
            raise ValueError("maxiter must be a positive integer.")
        tolerances = {}
        for name, value in (("gtol", gtol), ("ftol", ftol)):
            if np.ndim(value) != 0:
                raise ValueError(f"{name} must be a finite nonnegative real scalar.")
            scalar = _real_values(value, name).item()
            if scalar < 0:
                raise ValueError(f"{name} must be nonnegative.")
            tolerances[name] = scalar
        state = self.initial_state(initial_h)
        last = None

        def objective(z):
            nonlocal last
            evaluation = self.posterior.evaluate(z)
            last = evaluation
            return evaluation.potential, evaluation.gradient

        # A finite initial state is required. If its physical predictions are
        # unrepresentable, there is no preceding finite estimate to return.
        objective(state.coordinates["z"])
        try:
            result = minimize(
                objective,
                state.coordinates["z"],
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": int(maxiter), **tolerances},
            )
            # Line searches can finish after probing a point other than result.x.
            # Report the accepted terminal point rather than the final probe.
            if not np.array_equal(last.z, result.x):
                objective(result.x)
            success, message = bool(result.success), str(result.message)
        except NonfiniteProposal as exc:
            success = False
            message = f"Nonfinite optimization trial ({exc.stage}); returned the last finite state."
        return PosteriorEstimate(
            InferredState({"z": last.z}, self.problem_id),
            last.h,
            last.potential,
            success,
            message,
        )

    def close(self) -> None:
        r"""Release owned posterior workspaces, preserving the caller heat model.

        Returns
        -------
        None
            Repeated calls are harmless; the supplied forward model stays owned
            by its caller.
        """
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for posterior in getattr(self, "_workspaces", ()):
            posterior.close()
        self._workspaces = []

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        self.close()
