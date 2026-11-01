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

    def sample(
        self,
        *,
        initial=None,
        seed: int,
        warmup=1000,
        draws=2000,
        chains=4,
        output=None,
        controls=None,
        initial_states=None,
        progress=None,
    ):
        r"""Sample every conductivity degree of freedom in independent chains.

        Parameters
        ----------
        initial, initial_states : physical field or complete inferred states
            Supply exactly one common center or an explicit state per chain.
            A common center is dispersed only when more than one chain is used.
        seed : int
            Unsigned 32-bit master seed for independent threefry2x32 streams.
        warmup, draws, chains : int
            Per-chain warmup and retained counts; four independent chains default.
            Zero warmup requires disabled adaptation and explicit step size.
        output : path-like, optional
            New directory; defaults to a unique directory under ``runs``.
        controls : HMCControls, optional
            Frozen trajectory durations, spacing and initialization dispersion.
        progress : callable, optional
            Called after committed boundaries, with per-chain progress evidence.

        Returns
        -------
        PosteriorSamples
            Saved completed calculation. Failures leave inspectable evidence and
            propagate with its path. Caller conductivity/temperature are preserved.
        """
        self._require_open()
        if set(self._prior.coordinate_schema) != {"z"}:
            raise ValueError(
                "unsupported inferred coordinates for this Gaussian sampler"
            )
        if (initial is None) == (initial_states is None):
            raise ValueError("Supply exactly one of initial and initial_states.")
        from hmc_blackjax import HMCControls
        from hmc_run import RunRequest, prepare_initial_states, run_chains
        from posterior_store import RunStore
        from posterior_samples import PosteriorSamples
        from pathlib import Path
        from datetime import datetime, timezone
        from uuid import uuid4

        request = RunRequest(
            warmup,
            draws,
            chains,
            seed,
            controls if controls is not None else HMCControls(),
        )
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable.")
        if initial_states is None:
            state = self.initial_state(initial)
            supplied = [state] * request.chains
        else:
            supplied = list(initial_states)
            if len(supplied) != request.chains:
                raise ValueError("Explicit initial_states must match chains.")
            supplied = [self.initial_state(value) for value in supplied]
        actual, streams = prepare_initial_states(
            supplied, request, disperse=initial_states is None and request.chains > 1
        )
        record = self.to_record()
        metadata = request.metadata()
        metadata.update(
            initial_states=actual,
            random_streams=streams,
            initialization={
                "kind": (
                    "explicit_per_chain"
                    if initial_states is not None
                    else (
                        "dispersed_center"
                        if request.chains > 1
                        else "exact_single_state"
                    )
                ),
                "spread": request.controls.initial_spread,
                "scheme": "z_center + initial_spread * independent_standard_normal",
                "supplied_states": [state.coordinates for state in supplied],
            },
        )
        if output is None:
            parent = Path("runs")
            parent.mkdir(exist_ok=True)
            output = parent / (
                "hmc-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid4().hex
            )
        with RunStore.create(output, problem=record, run_metadata=metadata) as store:
            run_chains(
                self,
                initial_states=actual,
                request=request,
                store=store,
                progress=progress,
            )
        return PosteriorSamples.open(output)

    def resume(self, output, *, total_draws=None, progress=None):
        r"""Continue or extend the exact saved conductivity calculation.

        Parameters
        ----------
        output : path-like
            Existing completed or interrupted calculation directory.
        total_draws : int, optional
            Requested per-chain total, at least the original target. All existing
            warmup, retained fields, controls, starts and random states persist.
        progress : callable, optional
            Receives per-chain progress after each durable trajectory.

        Returns
        -------
        PosteriorSamples
            Completed calculation. Changed physics, sources or numerical
            environment fail before any new trajectory or target is written.
        """
        self._require_open()
        from hmc_run import RunRequest, run_chains
        from hmc_blackjax import require_sampling_environment
        from posterior_store import RunStore
        from posterior_samples import PosteriorSamples

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable.")
        require_sampling_environment()
        with RunStore.open(output, writable=True) as store:
            store.validate_identity(self.to_record())
            if total_draws is not None:
                store.update_target(total_draws)
            request = RunRequest.from_metadata(store.metadata)
            run_chains(
                self,
                initial_states=None,
                request=request,
                store=store,
                progress=progress,
            )
        return PosteriorSamples.open(output)

    def to_record(self):
        r"""Freeze all physical inputs and numerical provenance for reconstruction.

        This snapshot uses the owned frozen experiment, independent of subsequent
        changes to the caller's source, conductivity or temperature fields.

        Returns
        -------
        posterior_store.ProblemRecord
            Every mesh node, evaluated source, actual boundary temperature,
            measurement, Gaussian prior and fixed solver setting. Current caller
            conductivity and temperature are excluded from the physical law.
        """
        from dataclasses import asdict
        from posterior_store import ProblemRecord, numerical_environment

        self._require_open()
        model = self._model
        environment = numerical_environment()
        metadata = {
            "schema_version": 1,
            "model": {"cell_type": model.cell_type, "fingerprint": model.fingerprint},
            "field_order": "lexicographic_physical_x_y",
            "temperature_units": "K",
            "conductivity_definition": "h=exp(log(h_ref)+m)",
            "prior": self._prior.record(),
            "coordinate_schema": {
                name: list(shape)
                for name, shape in self._prior.coordinate_schema.items()
            },
            "numerics": asdict(self._numerics),
            "numerics_id": self.numerics_id,
            "numerical_environment": environment,
        }
        arrays = {
            name: getattr(model, name)
            for name in (
                "coordinates",
                "cells",
                "source",
                "boundary_dofs",
                "boundary_values",
            )
        }
        arrays.update(
            sensor_points=self._observations.points,
            sensor_values=self._observations.values,
            sensor_sigma=np.broadcast_to(
                self._observations.sigma, self._observations.values.shape
            ),
        )
        return ProblemRecord(
            metadata,
            arrays,
            self.problem_id,
            _digest({"solver": self.numerics_id, "environment": environment}),
        )

    @classmethod
    def from_record(cls, record):
        r"""Reconstruct the declared posterior, using only saved physical inputs.

        Parameters
        ----------
        record : posterior_store.ProblemRecord
            Frozen full-field experiment and fixed numerical settings.

        Returns
        -------
        HeatInverseProblem
            A problem with ``forward is None`` and independently owned heat and
            adjoint workspaces. No original Python callable or JAX is required.
        """
        from dolfinx import fem
        from posterior_store import ProblemRecord

        if not isinstance(record, ProblemRecord):
            raise TypeError("record must be a ProblemRecord.")
        # Own inputs even when a caller subsequently changes its record object.
        record = ProblemRecord(
            record.metadata, record.arrays, record.problem_id, record.numerical_id
        )
        meta, arrays = record.metadata, record.arrays
        if meta.get("schema_version") != 1:
            raise ValueError("Unsupported physical problem schema version.")
        if (
            meta.get("field_order") != "lexicographic_physical_x_y"
            or meta.get("temperature_units") != "K"
            or meta.get("conductivity_definition") != "h=exp(log(h_ref)+m)"
        ):
            raise ValueError("Unsupported physical units or field ordering.")
        required = {
            "coordinates",
            "cells",
            "source",
            "boundary_dofs",
            "boundary_values",
            "sensor_points",
            "sensor_values",
            "sensor_sigma",
        }
        if set(arrays) != required:
            raise ValueError("Physical problem arrays are incomplete or unsupported.")
        coordinates, cells = arrays["coordinates"], arrays["cells"]
        cell_type = meta["model"]["cell_type"]
        vertices = {"triangle": 3, "quadrilateral": 4}.get(cell_type)
        if (
            vertices is None
            or coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or len(coordinates) < vertices
        ):
            raise ValueError("Malformed physical mesh coordinates or cell type.")
        order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        if not np.array_equal(order, np.arange(len(order))) or len(
            np.unique(coordinates, axis=0)
        ) != len(order):
            raise ValueError(
                "Physical mesh nodes must be unique and in canonical order."
            )
        if (
            cells.dtype.kind not in "iu"
            or cells.ndim != 2
            or cells.shape[1] != vertices
            or len(cells) == 0
            or np.any(cells < 0)
            or np.any(cells >= len(coordinates))
        ):
            raise ValueError("Malformed physical mesh connectivity or indices.")
        if any(len(np.unique(cell)) != vertices for cell in cells):
            raise ValueError("A physical cell cannot repeat a vertex.")
        if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
            raise ValueError("Physical cells cannot be duplicated.")
        xy = coordinates[cells]
        u, v = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
        area = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        if np.any(area == 0):
            raise ValueError("Physical cells must have nonzero area.")
        if cell_type == "quadrilateral" and not np.allclose(
            xy[:, 0] + xy[:, 3], xy[:, 1] + xy[:, 2], rtol=0, atol=1e-14
        ):
            raise ValueError("Physical quadrilateral cells must be affine.")
        dofs = arrays["boundary_dofs"]
        if (
            dofs.dtype.kind not in "iu"
            or dofs.ndim != 1
            or not len(dofs)
            or np.any(dofs < 0)
            or np.any(dofs >= len(coordinates))
            or np.any(np.diff(dofs.astype(np.int64)) <= 0)
        ):
            raise ValueError("Malformed physical boundary indices.")
        if (
            arrays["source"].shape != (len(coordinates),)
            or arrays["boundary_values"].shape != dofs.shape
        ):
            raise ValueError(
                "Physical source/boundary field length disagrees with the mesh."
            )
        model = FrozenHeatModel(
            coordinates,
            cells,
            cell_type,
            arrays["source"],
            dofs,
            arrays["boundary_values"],
        )
        if model.fingerprint != meta["model"]["fingerprint"]:
            raise ValueError("Physical heat model identity mismatch.")
        observations = TemperatureObservations(
            arrays["sensor_points"], arrays["sensor_values"], arrays["sensor_sigma"]
        )
        numerics = HeatNumerics(**meta["numerics"])
        numerics_id = _digest(meta["numerics"])
        if numerics_id != meta["numerics_id"] or record.numerical_id != _digest(
            {"solver": numerics_id, "environment": meta["numerical_environment"]}
        ):
            raise ValueError("Numerical identity mismatch in physical problem record.")
        prior_record = meta["prior"]
        if prior_record.get("family") != "gaussian_log_conductivity":
            raise ValueError("Unsupported physical conductivity prior family.")
        if np.shape(prior_record["mu"]) != (len(coordinates),):
            raise ValueError("Prior mean must retain every physical node.")
        instance = cls.__new__(cls)
        instance._closed = False
        instance._workspaces = []
        instance._forward = None
        try:
            instance._model = model
            instance._observations = observations
            instance._numerics = numerics
            domain = model._build_mesh()
            V = fem.functionspace(domain, ("Lagrange", 1))
            prior = GaussianLogConductivity(
                mu=NodalField(prior_record["mu"], model.coordinates),
                kappa=prior_record["kappa"],
                tau=prior_record["tau"],
                h_ref=prior_record["h_ref"],
            )
            instance._prior = prior.bind(model, V)
            if instance._prior.record() != prior_record or meta[
                "coordinate_schema"
            ] != {
                name: list(shape)
                for name, shape in instance._prior.coordinate_schema.items()
            }:
                raise ValueError(
                    "Gaussian prior identity or complete coordinate schema mismatch."
                )
            instance._problem_id = _digest(
                {
                    "physical_model": model.fingerprint,
                    "sensor_points": observations.points.tolist(),
                    "sensor_values": observations.values.tolist(),
                    "sensor_sigma": np.broadcast_to(
                        observations.sigma, observations.values.shape
                    ).tolist(),
                    "prior": instance._prior.record(),
                    "coordinate_schema": instance._prior.coordinate_schema,
                }
            )
            if instance.problem_id != record.problem_id:
                raise ValueError("Physical posterior identity mismatch.")
            instance._numerics_id = numerics_id
            instance._posterior = instance._new_posterior(
                prefix=f"loaded_heat_inference_{next(_PREFIXES)}_"
            )
            return instance
        except BaseException:
            instance.close()
            raise

    @classmethod
    def load(cls, output):
        r"""Load the frozen physical posterior from a saved calculation directory.

        Parameters
        ----------
        output : path-like
            Directory created by ``RunStore.create``.

        Returns
        -------
        HeatInverseProblem
            Reconstructed posterior owning only private physical workspaces.
        """
        from posterior_store import RunStore

        with RunStore.open(output) as store:
            return cls.from_record(store.problem_record)

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
