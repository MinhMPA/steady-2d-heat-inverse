"""Numerical accuracy and same-discrete-model conductivity calibration studies.

The unit helpers do not certify a physical posterior. Scientific qualification
requires the preserved mesh comparisons, actual chains and 100 heat experiments.
Sampling, finite elements and plotting are imported only when requested.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

TOLERANCES = ((1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14))
SPREADS = (0.1, 0.3, 1.0)
RANK_POINTS = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))
QUANTILES = np.arange(1, 20, dtype=float) / 20


class _ScientificFailure(RuntimeError):
    """Irreparable numerical qualification evidence for this same prior draw."""


def _integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _encoded(value):
    return json.dumps(
        _json_value(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    """Atomically replace an advisory record; authoritative experiments persist."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_arrays(path, record_path, arrays, record):
    path, record_path = Path(path), Path(record_path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    record = {**record, "array_sha256": _file_hash(temporary)}
    _write(record_path, record)
    os.replace(temporary, path)
    return record


def _array_record(path, record_path):
    path, record_path = Path(path), Path(record_path)
    temporary = path.with_suffix(".tmp.npz")
    record = json.loads(record_path.read_text())
    candidate = path if path.exists() else temporary
    if not candidate.exists() or _file_hash(candidate) != record["array_sha256"]:
        raise ValueError(
            "Immutable saved field/data artifact changed or is incomplete."
        )
    if candidate == temporary:
        os.replace(temporary, path)
    return record


def _event(output, kind, **evidence):
    event = {"time_unix": time.time(), "kind": kind, **evidence}
    with (Path(output) / "events.jsonl").open("ab") as stream:
        stream.write(_encoded(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(_json_value(event)), flush=True)


@dataclass(frozen=True)
class StudyConfig:
    r"""Recorded physical choices and explicit numerical study controls.

    These parameters describe a demonstration, not inferred material constants.
    ``pilot_output`` locates the completed numerical qualification consumed by SBC.
    ``minimum_tolerance_index`` permits a newly declared tighter investigation;
    changed configurations require new output directories.
    """

    inference_meshes: tuple = (16, 32, 64)
    cell_type: str = "quadrilateral"
    source: float = 1.0
    bottom_temperature: float = 300.0
    sigma: float = 1e-3
    mu: float = math.log(2.0)
    kappa: float = 2.0
    tau: float = 1.0
    h_ref: float = 1.0
    conductivity_units: str = "declared conductivity units"
    seed: int = 20261026
    initial_step_size: float = 6.5e-4
    step_size_bounds: tuple = (1e-6, 1.3e-3)
    optimizer_maxiter: int = 5000
    optimizer_gtol: float = 1e-6
    optimizer_ftol: float = 1e-12
    minimum_tolerance_index: int = 0
    pilot_output: str = "runs/hmc-pilot"

    def __post_init__(self):
        object.__setattr__(self, "inference_meshes", tuple(self.inference_meshes))
        object.__setattr__(self, "step_size_bounds", tuple(self.step_size_bounds))
        if self.inference_meshes != (16, 32, 64) or self.cell_type != "quadrilateral":
            raise ValueError(
                "This qualification requires quadrilateral meshes (16,32,64)."
            )
        for name in (
            "source",
            "bottom_temperature",
            "sigma",
            "mu",
            "kappa",
            "tau",
            "h_ref",
            "initial_step_size",
            "optimizer_gtol",
            "optimizer_ftol",
        ):
            value = getattr(self, name)
            if np.ndim(value) != 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if (
            min(self.sigma, self.kappa, self.tau, self.h_ref, self.initial_step_size)
            <= 0
        ):
            raise ValueError(
                "Noise, prior scales and integration spacing must be positive."
            )
        if (
            len(self.step_size_bounds) != 2
            or not 0
            < self.step_size_bounds[0]
            <= self.initial_step_size
            <= self.step_size_bounds[1]
            <= math.pi / 8
        ):
            raise ValueError("Invalid integration-spacing bounds.")
        _integer(self.optimizer_maxiter, "optimizer_maxiter")
        _integer(self.seed, "seed", 0)
        if self.seed >= 2**32 or self.optimizer_gtol < 0 or self.optimizer_ftol < 0:
            raise ValueError("Invalid seed or optimization tolerances.")
        _integer(self.minimum_tolerance_index, "minimum_tolerance_index", 0)
        if self.minimum_tolerance_index >= len(TOLERANCES):
            raise ValueError("minimum_tolerance_index must be 0, 1 or 2.")
        if not self.pilot_output or not self.conductivity_units:
            raise ValueError("Pilot path and declared conductivity units are required.")

    @property
    def sensor_points(self):
        """The immutable physical 16-by-16 interior observation grid."""
        axis = np.arange(1, 17, dtype=float) / 17
        points = np.array([(x, y) for x in axis for y in axis])
        points.setflags(write=False)
        return points

    @classmethod
    def from_json(cls, path):
        r"""Read explicit demonstration choices from a JSON record."""
        return cls(**json.loads(Path(path).read_text()))


def sbc_ranks(truth, posterior_values, *, seed):
    r"""Count posterior values below each truth, randomizing exact tie positions.

    Parameters
    ----------
    truth : array-like
        Scalar truths with arbitrary leading shape.
    posterior_values : array-like
        Same leading shape followed by the retained rank-draw axis.
    seed : int
        Independent reproducible tie-breaking stream.

    Returns
    -------
    numpy.ndarray
        Integer ranks in 0,...,L, with the shape of ``truth``.
    """
    truth, draws = np.asarray(truth), np.asarray(posterior_values)
    if (
        draws.ndim != truth.ndim + 1
        or draws.shape[:-1] != truth.shape
        or draws.shape[-1] < 1
    ):
        raise ValueError(
            "Posterior values must have truth shape plus a nonempty draw axis."
        )
    if any(
        v.dtype.kind not in "fiu" or not np.isfinite(v).all() for v in (truth, draws)
    ):
        raise ValueError("Truths and posterior values must be finite real arrays.")
    below = np.sum(draws < truth[..., None], axis=-1)
    ties = np.sum(draws == truth[..., None], axis=-1)
    return below + np.random.default_rng(seed).integers(0, ties + 1)


def _cdf_deviation(ranks, *, posterior_count):
    ranks = np.asarray(ranks)
    _integer(posterior_count, "posterior_count")
    if (
        ranks.ndim != 1
        or not len(ranks)
        or ranks.dtype.kind not in "iu"
        or np.any((ranks < 0) | (ranks > posterior_count))
    ):
        raise ValueError(
            "Ranks must be a nonempty integer vector in 0,...,posterior_count."
        )
    return (ranks[:, None] <= np.arange(posterior_count + 1)).mean(axis=0) - np.arange(
        1, posterior_count + 2
    ) / (posterior_count + 1)


def sbc_uniform_envelope(*, replicates, posterior_count, quantities, seed):
    r"""Simulate a simultaneous discrete-CDF envelope for finite SBC experiments.

    Uses 100000 independent uniform-rank ensembles and a maximum absolute CDF
    deviation across all rank thresholds. Allocate 0.05/quantities probability
    to each reported physical quantity by Bonferroni. The Monte Carlo quantile
    uses the higher order statistic, preserving discreteness.

    Returns
    -------
    dict
        Threshold, lower/upper CDF bands and reproducibility information.
    """
    replicates = _integer(replicates, "replicates")
    posterior_count = _integer(posterior_count, "posterior_count")
    quantities = _integer(quantities, "quantities")
    rng = np.random.default_rng(seed)
    expected = np.arange(1, posterior_count + 2) / (posterior_count + 1)
    maxima = []
    for _ in range(100):
        ranks = rng.integers(0, posterior_count + 1, size=(1000, replicates))
        # Sorted samples avoid allocating ensemble x replication x threshold.
        counts = np.zeros((1000, posterior_count + 1), dtype=np.int64)
        np.add.at(counts, (np.repeat(np.arange(1000), replicates), ranks.ravel()), 1)
        cdf = np.cumsum(counts, axis=1) / replicates
        maxima.extend(np.max(np.abs(cdf - expected), axis=1))
    critical = float(np.quantile(maxima, 1 - 0.05 / quantities, method="higher"))
    return {
        "simulations": 100000,
        "seed": int(seed),
        "replicates": replicates,
        "posterior_count": posterior_count,
        "quantities": quantities,
        "alpha_per_quantity": 0.05 / quantities,
        "critical_deviation": critical,
        "expected": expected.tolist(),
        "lower": np.maximum(0, expected - critical).tolist(),
        "upper": np.minimum(1, expected + critical).tolist(),
        "method": "maximum absolute inclusive discrete-CDF deviation; Bonferroni",
    }


def _indicator_ess(values):
    from posterior_diagnostics import _load_backend, PACKAGE_VERSION

    backend, version = _load_backend()
    if version != PACKAGE_VERSION:
        raise RuntimeError(
            f"Quantile-indicator ESS requires arviz-stats=={PACKAGE_VERSION}; found {version}."
        )
    thresholds = np.quantile(values, QUANTILES, axis=(0, 1))
    result = np.empty((values.shape[-1], len(QUANTILES)))
    for q in range(values.shape[-1]):
        for j, threshold in enumerate(thresholds[:, q]):
            indicator = (values[:, :, q] <= threshold).astype(float)
            result[q, j] = backend.ess(
                indicator, method="mean", chain_axis=0, draw_axis=1
            )
    return result


def _autocorrelation(values, lag=1):
    # Never join chains. Preserve per-chain/quantity lag correlations.
    left, right = values[:, :-lag], values[:, lag:]
    a, b = left - left.mean(axis=1, keepdims=True), right - right.mean(
        axis=1, keepdims=True
    )
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return np.divide(
        np.sum(a * b, axis=1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )


def _rank_selection(values):
    """Choose 25 values per chain, with 19-indicator ESS and doubled spacing."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] != 4
        or values.shape[-1] != 5
        or values.shape[1] < 4
        or not np.isfinite(values).all()
    ):
        raise ValueError("Ranks require finite (4,draw,5) physical quantities.")
    raw_ess = _indicator_ess(values)
    if not np.isfinite(raw_ess).all() or np.min(raw_ess) <= 0:
        raise ValueError(
            "Raw quantile-indicator ESS is nonfinite or nonpositive; investigate."
        )
    raw_count = np.prod(values.shape[:2])
    prethin = 2 if np.any(raw_ess > raw_count) else 1
    work = values[:, ::prethin]
    ess = _indicator_ess(work) if prethin == 2 else raw_ess
    if not np.isfinite(ess).all() or np.min(ess) <= 0:
        raise ValueError(
            "Quantile-indicator ESS is missing or nonpositive; extend/investigate."
        )
    interval = prethin * max(1, int(np.ceil(np.prod(work.shape[:2]) / np.min(ess))))
    base = np.arange(25) * interval
    doubled = 2 * base
    result = {
        "interval": interval,
        "antithetic_prethin": prethin,
        "quantiles": QUANTILES.tolist(),
        "raw_quantile_ess": raw_ess.tolist(),
        "quantile_ess": ess.tolist(),
        "minimum_indicator_ess": float(np.min(ess)),
        "required_draws_per_chain": int(doubled[-1] + 1),
        "enough": bool(doubled[-1] < values.shape[1]),
    }
    if not result["enough"]:
        return result
    result["indices"] = np.tile(base, (4, 1)).tolist()
    result["double_indices"] = np.tile(doubled, (4, 1)).tolist()
    result["residual_autocorrelation"] = _json_value(
        _autocorrelation(values[:, ::interval])
    )
    result["double_residual_autocorrelation"] = _json_value(
        _autocorrelation(values[:, :: (2 * interval)])
    )
    result["selected_residual_autocorrelation"] = _json_value(
        _autocorrelation(values[:, base])
    )
    result["double_selected_residual_autocorrelation"] = _json_value(
        _autocorrelation(values[:, doubled])
    )
    # Quantile-indicator correlation is closer to the rank functional than h itself.
    threshold = np.quantile(values, QUANTILES, axis=(0, 1))
    result["indicator_residual_autocorrelation"] = [
        _json_value(_autocorrelation((values[:, ::interval] <= t).astype(float)))
        for t in threshold
    ]
    result["double_indicator_residual_autocorrelation"] = [
        _json_value(_autocorrelation((values[:, :: (2 * interval)] <= t).astype(float)))
        for t in threshold
    ]
    return result


def _study_manifest(config, output, *, kind, extra):
    from posterior_store import numerical_environment, source_revision

    output = Path(output).resolve()
    directory = Path(__file__).resolve().parent
    study_sources = {
        name: _file_hash(directory / (name + ".py"))
        for name in (
            "hmc_validation",
            "hmc_sbc_worker",
            "posterior_diagnostics",
            "posterior_samples",
        )
    }
    identity = {
        "schema": 1,
        "kind": kind,
        "config": asdict(config),
        "study_sources": study_sources,
        "environment": numerical_environment(),
        "extra": extra,
    }
    path = output / "manifest.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded.get("identity_sha256") != _hash(recorded["identity"]):
            raise ValueError("Saved study manifest identity checksum changed.")
        if recorded["identity"] != _json_value(identity):
            raise ValueError(
                "Study identity changed; use a different output directory."
            )
        return recorded
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "Existing study output has no manifest; use a different output directory."
        )
    manifest = {
        "identity": identity,
        "identity_sha256": _hash(identity),
        "source_revision": source_revision(),
        "created_unix": time.time(),
    }
    with path.open("xb") as stream:
        stream.write(_encoded(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _json_value(manifest)


@contextmanager
def _study_lock(output):
    with (Path(output) / "study.lock").open("a") as stream:
        os.set_inheritable(stream.fileno(), False)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "This physical study already has an active writer."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _seed(config, *labels):
    return int(np.random.SeedSequence([config.seed, *labels]).generate_state(1)[0])


def _observations(config, path=None):
    if path is None:
        if config.source != 1.0 or config.bottom_temperature != 300.0:
            raise ValueError(
                "Analytic demonstration data require source1 and bottom300; provide --data for another experiment."
            )
        points = config.sensor_points
        y = points[:, 1]
        values = config.bottom_temperature + (y - 0.5 * y * y) / 2
        values += np.random.default_rng(20261024).normal(0, config.sigma, len(y))
        return points, values, np.asarray(config.sigma)
    path = Path(path)
    if path.is_dir():
        from posterior_store import RunStore

        with RunStore.open(path) as saved:
            arrays = saved.problem_record.arrays
            result = tuple(
                arrays[name].copy()
                for name in ("sensor_points", "sensor_values", "sensor_sigma")
            )
    else:
        with np.load(path, allow_pickle=False) as archive:
            result = tuple(
                archive[name].copy() for name in ("points", "values", "sigma")
            )
    points, values, sigma = result
    if points.shape != config.sensor_points.shape or not np.array_equal(
        points, config.sensor_points
    ):
        raise ValueError(
            "Imported observations must retain the reference physical sensors."
        )
    if (
        values.shape != (256,)
        or not np.isfinite(values).all()
        or not np.all(sigma == config.sigma)
    ):
        raise ValueError(
            "Imported observations must be finite with the declared noise."
        )
    return result


@contextmanager
def _problem(config, nmesh, observations, tolerances, *, direct=False):
    from conductivity_prior import GaussianLogConductivity
    from forward_solver import SteadyHeat2DForwardSolver, _release_application_problem
    from heat_inference import HeatInverseProblem
    from heat_model import HeatNumerics
    from temperature_observations import TemperatureObservations

    numerics = HeatNumerics(
        rtol=tolerances[0],
        atol=tolerances[1],
        ksp_type="preonly" if direct else "cg",
        pc_type="lu" if direct else "hypre",
    )
    forward = SteadyHeat2DForwardSolver(
        nmesh=nmesh,
        mesh_type=config.cell_type,
        h=math.exp(math.log(config.h_ref) + config.mu),
        q=config.source,
        DBC_value=config.bottom_temperature,
        petsc_opts=numerics.petsc_options(),
    )
    try:
        with HeatInverseProblem(
            forward=forward,
            observations=TemperatureObservations(*observations),
            prior=GaussianLogConductivity(
                mu=config.mu, kappa=config.kappa, tau=config.tau, h_ref=config.h_ref
            ),
            numerics=numerics,
        ) as problem:
            yield problem
    finally:
        _release_application_problem(forward.problem)


def _optimize(config, problem, directory):
    from heat_inference import InferredState

    path = Path(directory) / "estimate.npz"
    info_path = Path(directory) / "optimization.json"
    if path.exists() or (path.with_suffix(".tmp.npz").exists() and info_path.exists()):
        info = _array_record(path, info_path)
        with np.load(path, allow_pickle=False) as archive:
            z = archive["z"].copy()
        return InferredState({"z": z}, problem.problem_id), info
    started = time.monotonic()
    before = asdict(problem.posterior.solve_counts)
    median = math.exp(math.log(config.h_ref) + config.mu)
    estimate = problem.optimize(
        initial_h=median,
        maxiter=config.optimizer_maxiter,
        gtol=config.optimizer_gtol,
        ftol=config.optimizer_ftol,
    )
    evaluation = problem.posterior.evaluate(estimate.state.coordinates["z"])
    after = asdict(problem.posterior.solve_counts)
    info = {
        "success": estimate.success,
        "message": estimate.message,
        "initial_h": median,
        "maxiter": config.optimizer_maxiter,
        "gtol": config.optimizer_gtol,
        "ftol": config.optimizer_ftol,
        "seconds": time.monotonic() - started,
        "potential": evaluation.potential,
        "phi": evaluation.phi,
        "gradient_norm": float(np.linalg.norm(evaluation.gradient)),
        "solve_counts": {name: after[name] - before[name] for name in before},
        "interpretation": (
            "optimizer-converged mode candidate"
            if estimate.success
            else "finite unconverged mode candidate"
        ),
    }
    # The complete finite state is lawful even when optimization reports failure.
    info = _save_arrays(
        path,
        info_path,
        {"z": estimate.state.coordinates["z"], "h": estimate.conductivity},
        info,
    )
    return estimate.state, info


def _force_comparison(config, nmesh, observations, positions, labels, output):
    """Identical whitened fields and data, with independent direct-LU evaluation."""
    rows, costs = [], []
    reference_started = time.monotonic()
    with _problem(
        config, nmesh, observations, TOLERANCES[-1], direct=True
    ) as reference:
        truth = [reference.posterior.evaluate(z) for z in positions]
        reference_cost = {
            "seconds": time.monotonic() - reference_started,
            "solve_counts": asdict(reference.posterior.solve_counts),
        }
        _event(
            output,
            "direct_lu_comparison",
            nmesh=nmesh,
            states=len(positions),
            **reference_cost,
        )
        for index, tolerance in enumerate(TOLERANCES):
            if index < config.minimum_tolerance_index:
                continue
            before_time = time.monotonic()
            with _problem(config, nmesh, observations, tolerance) as candidate:
                for label, z, ref in zip(labels, positions, truth):
                    evaluation = candidate.posterior.evaluate(z)
                    energy = abs(evaluation.potential - ref.potential)
                    force = np.linalg.norm(evaluation.gradient - ref.gradient) / max(
                        1, np.linalg.norm(ref.gradient)
                    )
                    likelihood = np.linalg.norm(
                        evaluation.grad_phi - ref.grad_phi
                    ) / max(1, np.linalg.norm(ref.grad_phi))
                    rows.append(
                        {
                            "label": label,
                            "tolerance_index": index,
                            "rtol": tolerance[0],
                            "atol": tolerance[1],
                            "absolute_delta_U": float(energy),
                            "relative_grad_U_error": float(force),
                            "relative_grad_phi_error": float(likelihood),
                            "passes": bool(
                                energy < 1e-4 and force < 1e-5 and likelihood < 1e-5
                            ),
                        }
                    )
                cost = {
                    "tolerance_index": index,
                    "seconds": time.monotonic() - before_time,
                    "solve_counts": asdict(candidate.posterior.solve_counts),
                }
                costs.append(cost)
                _event(
                    output,
                    "force_comparison",
                    nmesh=nmesh,
                    tolerance=tolerance,
                    states=len(positions),
                    **cost,
                )
    passing = [
        index
        for index in range(config.minimum_tolerance_index, len(TOLERANCES))
        if all(row["passes"] for row in rows if row["tolerance_index"] == index)
    ]
    return {
        "comparisons": rows,
        "passing_tolerance_indices": passing,
        "reference_cost": reference_cost,
        "candidate_costs": costs,
    }


def _controls(config, spread):
    from hmc_blackjax import HMCControls

    return HMCControls(
        initial_step_size=config.initial_step_size,
        step_size_bounds=config.step_size_bounds,
        initial_spread=spread,
    )


def _chain_evidence(path, lower_bound):
    from posterior_store import RunStore

    with RunStore.open(path) as store:
        chains = []
        for chain in range(store.metadata["chains"]):
            initial = store.initial_checkpoint(chain)
            checkpoint = store.checkpoint(chain)
            warmup = list(store.records(chain, phase="warmup"))
            retained = list(store.records(chain))
            statistics = [r.statistics for r in retained]
            tail = warmup[-64:]
            residual = (
                initial.physical_state["sensor_temperature"]
                - store.problem_record.arrays["sensor_values"]
            ) / store.problem_record.arrays["sensor_sigma"]
            initial_phi = float(0.5 * residual @ residual)
            item = {
                "chain": chain,
                "initial_phi": initial_phi,
                "warmup_count": len(warmup),
                "retained_count": len(retained),
                "final_warmup_divergences": sum(
                    r.statistics["divergent"] for r in tail
                ),
                "retained_divergences": sum(
                    r.statistics["divergent"] for r in retained
                ),
                "mean_retained_acceptance": (
                    float(np.mean([s["acceptance_probability"] for s in statistics]))
                    if statistics
                    else None
                ),
                "minimum_step_size": min(
                    [r.statistics["step_size"] for r in warmup + retained],
                    default=lower_bound,
                ),
                "final_step_size": checkpoint.step_size,
                "solve_counts": checkpoint.counters,
                "statistics": {
                    "warmup": [r.statistics for r in warmup],
                    "retained": statistics,
                },
            }
            item["passes_spread_criteria"] = bool(
                len(warmup) == 256
                and len(retained) == 256
                and item["final_warmup_divergences"] == 0
                and item["retained_divergences"] == 0
                and item["minimum_step_size"] > lower_bound
                and item["final_step_size"] > lower_bound
                and item["mean_retained_acceptance"] >= 0.65
                and not [
                    f
                    for f in store.failures()
                    if f.get("exception")
                    not in ("KeyboardInterrupt", "InterruptedError")
                ]
            )
            chains.append(item)
        return {
            "chains": chains,
            "failures": list(store.failures()),
            "passes_spread_criteria": all(
                item["passes_spread_criteria"] for item in chains
            ),
            "seed": store.metadata["seed"],
            "random_streams": store.metadata["random_streams"],
        }


def _sample(
    config,
    problem,
    state,
    path,
    output,
    *,
    seed,
    spread,
    warmup,
    draws,
    cancel_check=None,
):
    from posterior_samples import PosteriorSamples

    path = Path(path).resolve()

    def progress(item):
        if cancel_check is not None:
            cancel_check(f"committed:{item.chain}:{item.phase}:{item.completed}")
        if (
            item.completed == 1
            or item.completed % 16 == 0
            or item.completed == item.target
        ):
            _event(
                output,
                "trajectory_progress",
                run=str(path),
                chain=item.chain,
                phase=item.phase,
                completed=item.completed,
                target=item.target,
                seconds=item.elapsed_seconds,
            )

    output = Path(output).resolve()
    if cancel_check is not None:
        cancel_check("before_sampling_segment")
    started = time.monotonic()
    try:
        if path.exists():
            with PosteriorSamples.open(path) as prior:
                fatal = [
                    f
                    for f in prior._store.failures()
                    if f.get("exception")
                    not in ("KeyboardInterrupt", "InterruptedError")
                ]
                if fatal:
                    raise RuntimeError(
                        "Saved posterior has a failure requiring investigation: "
                        + str(path)
                    )
                target = max(draws, prior.metadata["total_draws"])
            samples = problem.resume(path, total_draws=target, progress=progress)
        else:
            samples = problem.sample(
                initial=state,
                seed=seed,
                chains=4,
                warmup=warmup,
                draws=draws,
                controls=_controls(config, spread),
                output=path,
                progress=progress,
            )
        _event(
            output,
            "sampling_segment_complete",
            run=str(path),
            retained_per_chain=samples.metadata["total_draws"],
        )
        return samples
    finally:
        _event(
            output,
            "sampling_segment_elapsed",
            run=str(path),
            seconds=time.monotonic() - started,
        )


def _sampling_seconds(output, path):
    events = Path(output) / "events.jsonl"
    if not events.exists():
        return 0.0
    return sum(
        item["seconds"]
        for item in map(json.loads, events.read_text().splitlines())
        if item["kind"] == "sampling_segment_elapsed"
        and Path(item["run"]).resolve() == Path(path).resolve()
    )


def _minimum_ess(report):
    numbers = []
    for quantity in report.per_quantity.values():
        fixed = np.asarray(quantity["analytically_fixed"], dtype=bool)
        for name in ("bulk_ess", "tail_ess"):
            numbers.extend(np.asarray(quantity[name], dtype=float)[~fixed].ravel())
    return float(np.min(numbers)) if numbers and np.isfinite(numbers).all() else None


def _posterior_states(samples, count=10):
    counts = samples._counts()
    if len(counts) != 4 or min(counts) < 3:
        raise ValueError(
            "Ten posterior force states require actual four-chain retained draws."
        )
    indices = [
        (i % 4, int((i // 4 + 1) * int(counts[i % 4]) / 4)) for i in range(count)
    ]
    return [
        samples.state(chain=c, draw=d).coordinates["z"] for c, d in indices
    ], indices


def _quantity_operator(record):
    from posterior_diagnostics import _physical_observables

    weights, _ = _physical_observables(record)
    coordinates = record.arrays["coordinates"]
    nodes = []
    for point in RANK_POINTS:
        found = np.flatnonzero(
            np.all(np.isclose(coordinates, point, rtol=0, atol=1e-13), axis=1)
        )
        if len(found) != 1:
            raise ValueError(
                "Rank-monitor location must be one unique conductivity node."
            )
        nodes.append(int(found[0]))
    operator = np.zeros((5, len(coordinates)))
    operator[0] = weights[0]
    operator[np.arange(1, 5), nodes] = 1.0
    return operator


def _quantities(record, fields):
    return np.asarray(fields) @ _quantity_operator(record).T


def _saved_quantities(samples):
    operator = _quantity_operator(samples._record)
    counts = samples._counts()
    if len(counts) != 4 or len(set(counts)) != 1:
        raise ValueError("Rank quantities require four equally complete chains.")
    values = np.empty((4, int(counts[0]), 5))
    for chain in range(4):
        for draw, row in enumerate(samples._store.records(chain)):
            values[chain, draw] = row.h @ operator.T
    return values


def _checksums(directory):
    paths = ("problem.json", "problem.npz", "chains.sqlite")
    return {name: _file_hash(Path(directory) / name) for name in paths}


def _require_native(manifest):
    evidence = manifest["identity"]["environment"]["native_builds"]
    if any(item["status"] != "loaded" for item in evidence["imports"].values()):
        raise RuntimeError(
            "Study requires inspected native FEM, MPI and numerical libraries."
        )
    if not evidence["libraries"] or any(
        item["status"] != "inspected" for item in evidence["compiler"].values()
    ):
        raise RuntimeError(
            "Study requires actual compiler and native library provenance."
        )


def _failure(output, path, exc, **context):
    evidence = {"exception": type(exc).__name__, "message": str(exc), **context}
    for name in ("stage", "position", "reason", "residual_norm", "iterations"):
        if hasattr(exc, name):
            evidence[name] = getattr(exc, name)
    _write(path, evidence)
    _event(output, "failure", **evidence)


def run_numerical_pilot(config, output, *, data_path=None):
    r"""Measure fixed-solve accuracy, initialization spreads and actual pilot cost.

    Runs all three n16 spread pilots with four chains, 256 warmup and 256 draws.
    Meshes 32 and 64 use the selected spread and the same pilot lengths for ten
    retained-state force checks. These pilots are not convergence evidence.
    Existing output resumes the recorded calculation; scientific failures require
    investigation and a newly declared experiment before settings can change.

    Returns
    -------
    dict
        Actual mesh-wise comparisons, spread criteria and initial production plan.
    """
    if not isinstance(config, StudyConfig):
        raise TypeError("config must be StudyConfig.")
    output = Path(output).resolve()
    stored_data = output / "observations.npz"
    observation = _observations(
        config,
        (
            data_path
            if data_path is not None
            else (stored_data if stored_data.exists() else None)
        ),
    )
    observation_hash = _hash([array.tolist() for array in observation])
    manifest = _study_manifest(
        config, output, kind="pilot", extra={"observation_sha256": observation_hash}
    )
    _require_native(manifest)
    with _study_lock(output):
        if not stored_data.exists():
            np.savez(
                stored_data,
                points=observation[0],
                values=observation[1],
                sigma=observation[2],
            )
        if (output / "qualification.json").exists():
            qualification = json.loads((output / "qualification.json").read_text())
            # Immutable native chain digests are checked before reuse.
            for mesh in qualification["meshes"].values():
                for run in mesh["runs"]:
                    if (
                        "checksums" in run
                        and _checksums(output / run["relative_path"])
                        != run["checksums"]
                    ):
                        raise ValueError(
                            "Qualified pilot chains changed after their recorded checksums."
                        )
            return qualification
        if (output / "failure.json").exists():
            raise RuntimeError(
                "Recorded pilot failure requires investigation; it will not be overwritten."
            )
        meshes = {}
        spread = None
        production_length = None
        try:
            for nmesh in config.inference_meshes:
                directory = output / f"mesh-{nmesh}"
                directory.mkdir(exist_ok=True)
                with _problem(
                    config, nmesh, observation, TOLERANCES[-1], direct=True
                ) as reference:
                    state, optimization = _optimize(config, reference, directory)
                    if not optimization["success"]:
                        raise RuntimeError(
                            f"Mesh {nmesh} mode optimization did not converge; investigate recorded candidate."
                        )
                    mode = state.coordinates["z"]
                    rng = np.random.default_rng(_seed(config, 10, nmesh))
                    positions = [mode] + [
                        mode + rng.normal(size=len(mode)) for _ in range(10)
                    ]
                    field_path = directory / "force_fields.npz"
                    if field_path.exists():
                        with np.load(field_path, allow_pickle=False) as fields:
                            if not np.array_equal(
                                fields["positions"], np.asarray(positions)
                            ):
                                raise ValueError(
                                    "Saved preliminary force fields changed."
                                )
                    else:
                        np.savez(field_path, positions=np.asarray(positions))
                comparison_path = directory / "preliminary-forces.json"
                if comparison_path.exists():
                    preliminary = json.loads(comparison_path.read_text())
                else:
                    preliminary = _force_comparison(
                        config,
                        nmesh,
                        observation,
                        positions,
                        ["mode"] + [f"prior-correlated-{i}" for i in range(10)],
                        output,
                    )
                    _write(comparison_path, preliminary)
                if not preliminary["passing_tolerance_indices"]:
                    raise RuntimeError(
                        f"No fixed heat tolerances pass at mesh {nmesh}."
                    )
                tolerance_index = min(preliminary["passing_tolerance_indices"])
                tolerance = TOLERANCES[tolerance_index]
                trial_spreads = SPREADS if nmesh == 16 else (spread,)
                runs = []
                for spread_index, trial_spread in enumerate(trial_spreads):
                    run_path = directory / f"spread-{trial_spread:g}"
                    seed = _seed(config, 20, nmesh, SPREADS.index(trial_spread))
                    evidence_path = directory / f"spread-{trial_spread:g}-evidence.json"
                    if evidence_path.exists():
                        evidence = json.loads(evidence_path.read_text())
                        if (
                            "checksums" in evidence
                            and _checksums(run_path) != evidence["checksums"]
                        ):
                            raise ValueError(
                                "Pilot chain changed after recorded evidence."
                            )
                        runs.append(evidence)
                        continue
                    try:
                        with _problem(config, nmesh, observation, tolerance) as problem:
                            from heat_inference import InferredState

                            initial = InferredState({"z": mode}, problem.problem_id)
                            with _sample(
                                config,
                                problem,
                                initial,
                                run_path,
                                output,
                                seed=seed,
                                spread=trial_spread,
                                warmup=256,
                                draws=256,
                            ) as samples:
                                diagnostics = samples.diagnostics()
                                evidence = _chain_evidence(
                                    run_path, config.step_size_bounds[0]
                                )
                                evidence.update(
                                    spread=trial_spread,
                                    sampling_seconds=_sampling_seconds(
                                        output, run_path
                                    ),
                                    minimum_ess=_minimum_ess(diagnostics),
                                    diagnostics=asdict(diagnostics),
                                    relative_path=str(run_path.relative_to(output)),
                                )
                        evidence["checksums"] = _checksums(run_path)
                    except (KeyboardInterrupt, InterruptedError):
                        raise
                    except Exception as exc:
                        _failure(
                            output,
                            directory / f"spread-{trial_spread:g}-failure.json",
                            exc,
                            nmesh=nmesh,
                            spread=trial_spread,
                            seed=seed,
                        )
                        evidence = {
                            "spread": trial_spread,
                            "passes_spread_criteria": False,
                            "relative_path": str(run_path.relative_to(output)),
                            "failure": {
                                "exception": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                        if (run_path / "chains.sqlite").exists():
                            evidence["checksums"] = _checksums(run_path)
                    _write(evidence_path, evidence)
                    runs.append(evidence)
                eligible = [r for r in runs if r["passes_spread_criteria"]]
                if not eligible:
                    raise RuntimeError(
                        f"No tested initialization spread qualifies at mesh {nmesh}."
                    )
                chosen = max(eligible, key=lambda r: r["spread"])
                if nmesh == 16:
                    spread = chosen["spread"]
                    ess = chosen["minimum_ess"]
                    if ess is None or ess <= 0:
                        raise RuntimeError(
                            "Pilot ESS is not finite and positive; cannot estimate production cost."
                        )
                    production_length = max(2000, 2 * math.ceil(256 * 400 / ess))
                from posterior_samples import PosteriorSamples

                with PosteriorSamples.open(output / chosen["relative_path"]) as samples:
                    positions, indices = _posterior_states(samples)
                posterior_comparison = _force_comparison(
                    config,
                    nmesh,
                    observation,
                    positions,
                    [f"chain-{c}-draw-{d}" for c, d in indices],
                    output,
                )
                _write(directory / "posterior-forces.json", posterior_comparison)
                if (
                    tolerance_index
                    not in posterior_comparison["passing_tolerance_indices"]
                ):
                    raise RuntimeError(
                        f"Provisional tolerances fail posterior-state forces at mesh {nmesh}; a new numerical identity/run is required."
                    )
                meshes[str(nmesh)] = {
                    "nmesh": nmesh,
                    "dimension": len(mode),
                    "optimization": optimization,
                    "preliminary_forces": preliminary,
                    "posterior_forces": posterior_comparison,
                    "posterior_state_indices": indices,
                    "tolerance_index": tolerance_index,
                    "tolerances": tolerance,
                    "spread": chosen["spread"],
                    "runs": runs,
                    "status": "numerically_qualified_pilot_only",
                }
                _write(
                    output / "progress.json",
                    {"meshes": meshes, "status": "pilot_in_progress"},
                )
            qualification = {
                "status": "numerical_pilot_complete",
                "manifest_sha256": manifest["identity_sha256"],
                "meshes": meshes,
                "selected_spread": spread,
                "production_warmup": 1000,
                "initial_production_draws": production_length,
                "estimated_production_seconds": next(
                    r["sampling_seconds"]
                    for r in meshes["16"]["runs"]
                    if r["spread"] == spread
                )
                * (1000 + production_length)
                / 512,
                "timing_caveat": "Linear extrapolation from actual four-chain pilot elapsed time; actual diagnostics determine completion.",
                "length_rule": "max(2000, 2*ceil(256*400/min_pilot_ESS)); confirm through actual diagnostics",
                "sampling_convergence": "not established by pilots",
                "calibration": "not yet assessed",
                "mesh_approximation": "not assessed",
                "sharp_truth_prior_adequacy": "not assessed",
            }
            _write(output / "qualification.json", qualification)
            _event(
                output,
                "numerical_pilot_complete",
                spread=spread,
                production_draws=production_length,
            )
            return _json_value(qualification)
        except (KeyboardInterrupt, InterruptedError):
            _event(output, "interrupted", stage="pilot")
            raise
        except Exception as exc:
            _failure(
                output,
                output / "failure.json",
                exc,
                stage="pilot",
                completed_meshes=list(meshes),
            )
            raise


def _qualified_pilot(config):
    path = Path(config.pilot_output).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    # Validate the exact environment/configuration by the same continuation check.
    current = _study_manifest(
        config, path, kind="pilot", extra=manifest["identity"]["extra"]
    )
    _require_native(current)
    qualification = json.loads((path / "qualification.json").read_text())
    if (
        qualification["status"] != "numerical_pilot_complete"
        or qualification["manifest_sha256"] != manifest["identity_sha256"]
    ):
        raise ValueError(
            "SBC requires completed matching numerical pilot qualification."
        )
    for mesh in qualification["meshes"].values():
        for run in mesh["runs"]:
            if (
                "checksums" in run
                and _checksums(path / run["relative_path"]) != run["checksums"]
            ):
                raise ValueError("Numerical pilot data changed after qualification.")
    return qualification, _file_hash(path / "qualification.json")


def _extension_target(report, current):
    if report.divergences:
        raise _ScientificFailure(
            "Retained divergences cannot be cured by extension; investigate saved posterior."
        )
    minimum = _minimum_ess(report)
    if (
        minimum is None
        or minimum <= 0
        or any(
            token in reason.lower()
            for reason in report.reasons
            for token in ("nonfinite", "missing", "accidentally constant")
        )
    ):
        raise _ScientificFailure(
            "Missing/nonfinite convergence evidence requires investigation, not blind extension."
        )
    return max(2 * current, 2 * math.ceil(current * 400 / minimum))


def _converged_samples(
    config, problem, initial, path, output, *, seed, spread, draws, cancel_check=None
):
    """Extend the same chains until ordinary qualification; preserve every draw."""
    while True:
        samples = _sample(
            config,
            problem,
            initial,
            path,
            output,
            seed=seed,
            spread=spread,
            warmup=1000,
            draws=draws,
            cancel_check=cancel_check,
        )
        try:
            if cancel_check is not None:
                cancel_check("before_convergence_diagnostics")
            report = samples.diagnostics()
            if any(
                samples._store.checkpoint(chain).step_size <= config.step_size_bounds[0]
                or any(
                    row.statistics["step_size"] <= config.step_size_bounds[0]
                    for row in samples._store.records(chain)
                )
                for chain in range(4)
            ):
                raise _ScientificFailure(
                    "Integration spacing reached its lower bound; investigate the preserved posterior."
                )
            _write(Path(path).parent / "convergence.json", asdict(report))
            if report.qualified:
                return samples, report
            current = samples.metadata["total_draws"]
            minimum = _minimum_ess(report)
            draws = _extension_target(report, current)
            _event(
                output,
                "extend_for_convergence",
                run=str(path),
                current_draws=current,
                minimum_ess=minimum,
                new_total=draws,
                reasons=report.reasons,
            )
        except BaseException:
            samples.close()
            raise
        samples.close()


def _calibration_figures(output, ranks, doubled, envelope):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Area mean h",
        "h(0.25,0.25)",
        "h(0.25,0.75)",
        "h(0.75,0.25)",
        "h(0.75,0.75)",
    ]
    figure, axes = plt.subplots(5, 2, figsize=(10, 16), constrained_layout=True)
    expected = np.asarray(envelope["expected"])
    for quantity, label in enumerate(labels):
        axes[quantity, 0].hist(
            ranks[:, quantity],
            bins=np.arange(-0.5, 101.5, 1),
            histtype="step",
            label="chosen interval",
        )
        axes[quantity, 0].hist(
            doubled[:, quantity],
            bins=np.arange(-0.5, 101.5, 1),
            histtype="step",
            label="doubled interval",
        )
        axes[quantity, 0].axhline(len(ranks) / 101, color="0.5", linewidth=0.8)
        axes[quantity, 0].set(xlabel="Posterior rank", ylabel="Count", title=label)
        x = np.arange(101)
        axes[quantity, 1].fill_between(
            x,
            np.asarray(envelope["lower"]) - expected,
            np.asarray(envelope["upper"]) - expected,
            color="0.85",
            label="95% familywise CDF envelope",
        )
        axes[quantity, 1].step(
            x, _cdf_deviation(ranks[:, quantity], posterior_count=100), where="post"
        )
        axes[quantity, 1].step(
            x,
            _cdf_deviation(doubled[:, quantity], posterior_count=100),
            where="post",
            linestyle="--",
        )
        axes[quantity, 1].axhline(0, color="0.5", linewidth=0.8)
        axes[quantity, 1].set(
            xlabel="Posterior rank", ylabel="Empirical CDF minus uniform CDF"
        )
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.savefig(Path(output) / "rank-calibration.pdf")
    figure.savefig(Path(output) / "rank-calibration.png", dpi=160)
    plt.close(figure)


def _sbc_seeds(config, index):
    return {
        name: _seed(config, 30, index, stream)
        for stream, name in enumerate(("truth", "noise", "sampling", "rank_ties"))
    }


def _checked_artifact(path, record_path):
    """Validate committed or recoverable arrays without a parent-side rename."""
    path, record_path = Path(path), Path(record_path)
    info = json.loads(record_path.read_text())
    candidate = path if path.exists() else path.with_suffix(".tmp.npz")
    if not candidate.exists() or _file_hash(candidate) != info["array_sha256"]:
        raise ValueError("Incomplete or changed scientific array artifact.")
    return info


def _validate_resume_artifacts(directory, seeds=None):
    from posterior_store import RunStore

    directory = Path(directory)
    evidence = {}
    for array, record in (
        ("generated.npz", "generation.json"),
        ("estimate.npz", "optimization.json"),
    ):
        path = directory / array
        record_path = directory / record
        if (
            path.exists()
            or path.with_suffix(".tmp.npz").exists()
            or record_path.exists()
        ):
            info = _checked_artifact(path, record_path)
            evidence[record] = info["array_sha256"]
            if (
                record == "generation.json"
                and seeds is not None
                and info["seeds"] != seeds
            ):
                raise ValueError(
                    "Saved generation streams differ from the fixed index."
                )
    posterior = directory / "posterior"
    if posterior.exists():
        with RunStore.open(posterior) as store:
            metadata = store.metadata
            if metadata["chains"] != 4 or (
                seeds is not None and metadata["seed"] != seeds["sampling"]
            ):
                raise ValueError("Saved chain count or sampling stream changed.")
            fatal = [
                item
                for item in store.failures()
                if item.get("exception")
                not in ("KeyboardInterrupt", "InterruptedError")
            ]
            if fatal:
                raise ValueError("Saved posterior failure requires investigation.")
            checkpoints = [store.checkpoint(chain) for chain in range(4)]
            evidence["target_draws"] = metadata["total_draws"]
            evidence["committed"] = [
                {
                    "chain": cp.chain,
                    "warmup_done": cp.warmup_done,
                    "draws_done": cp.draws_done,
                    "phase": cp.phase,
                }
                for cp in checkpoints
            ]
    return evidence


def _validate_sbc_result(config, manifest, directory, result):
    index = _integer(result["replicate"], "replicate", 0)
    total = manifest["identity"]["extra"]["replicates"]
    if (
        index >= total
        or result["replicates"] != total
        or result["identity"] != manifest["identity_sha256"]
        or result["seeds"] != _sbc_seeds(config, index)
    ):
        raise ValueError("Replica index, identity or scientific streams changed.")
    directory = Path(directory)
    if result["status"] == "qualified":
        generation = _checked_artifact(
            directory / "generated.npz", directory / "generation.json"
        )
        if (
            generation["seeds"] != result["seeds"]
            or generation["array_sha256"] != result["generation_sha256"]
            or _checksums(directory / "posterior") != result["checksums"]
        ):
            raise ValueError(
                "Saved calibrated posterior or generated observations changed."
            )
        for name in ("ranks", "doubled_interval_ranks"):
            ranks = np.asarray(result[name])
            if (
                ranks.shape != (5,)
                or ranks.dtype.kind not in "iu"
                or np.any((ranks < 0) | (ranks > 100))
            ):
                raise ValueError("Saved ranks do not match the declared experiment.")
    elif result["status"] == "failed":
        if json.loads((directory / "failure.json").read_text()) != result["failure"]:
            raise ValueError("Saved scientific failure evidence changed.")
    else:
        raise ValueError("Unknown terminal scientific status.")
    return result


def _worker_replica(output, *, index, replicates, identity, cancel_check):
    """Validate the existing study under worker ownership, then do one index."""
    output = Path(output).resolve()
    saved = json.loads((output / "manifest.json").read_text())
    config = StudyConfig(**saved["identity"]["config"])
    cancel_check("before_native_verification")
    pilot, pilot_hash = _qualified_pilot(config)
    manifest = _study_manifest(
        config,
        output,
        kind="sbc",
        extra={"replicates": replicates, "pilot_qualification_sha256": pilot_hash},
    )
    _require_native(manifest)
    if manifest["identity_sha256"] != identity or not 0 <= index < replicates:
        raise ValueError("Issued worker does not match the fixed study.")
    return _run_sbc_replica(
        config,
        output,
        index=index,
        replicates=replicates,
        pilot=pilot,
        cancel_check=cancel_check,
    )


def _run_sbc_replica(config, output, *, index, replicates, pilot, cancel_check):
    """The single scientific routine used by serial and concurrent fresh workers."""
    from conductivity_posterior import HeatSolveFailure, NonfiniteProposal

    index = _integer(index, "replicate", 0)
    if index >= replicates:
        raise ValueError("Replica index lies outside the fixed study.")
    output = Path(output).resolve()
    directory = output / f"replicate-{index:03d}"
    directory.mkdir(exist_ok=True)
    replicate = index
    seeds = _sbc_seeds(config, index)
    tolerance = tuple(pilot["meshes"]["16"]["tolerances"])
    spread = pilot["selected_spread"]
    data_path = directory / "generated.npz"
    cancel_check("before_generation")
    try:
        generation_path = directory / "generation.json"
        if data_path.with_suffix(".tmp.npz").exists() and generation_path.exists():
            _array_record(data_path, generation_path)
        if not data_path.exists():
            empty = (
                config.sensor_points,
                np.full(256, config.bottom_temperature),
                np.asarray(config.sigma),
            )
            with _problem(config, 16, empty, tolerance) as generator:
                z = np.random.default_rng(seeds["truth"]).normal(
                    size=generator.posterior.prior.dimension
                )
                h = generator.posterior.prior.to_h(z)
                _, prediction = generator.posterior.predict_h(h)
                values = prediction + np.random.default_rng(seeds["noise"]).normal(
                    0, config.sigma, 256
                )
                truth = _quantities(generator.to_record(), h)
                generating_problem_id = generator.problem_id
            _save_arrays(
                data_path,
                generation_path,
                {
                    "z": z,
                    "h": h,
                    "truth": truth,
                    "points": config.sensor_points,
                    "values": values,
                    "sigma": np.asarray(config.sigma),
                    "prediction": prediction,
                },
                {
                    "seeds": seeds,
                    "nmesh": 16,
                    "generating_problem_id": generating_problem_id,
                    "tolerances": tolerance,
                    "truth_law": "same discrete Gaussian prior and heat model as inference",
                },
            )
        cancel_check("after_generation")
        generation = _array_record(data_path, generation_path)
        if generation["seeds"] != seeds:
            raise ValueError("Generated truth/data random streams changed.")
        with np.load(data_path, allow_pickle=False) as archive:
            observation = tuple(
                archive[name].copy() for name in ("points", "values", "sigma")
            )
            truth = archive["truth"].copy()
        with _problem(config, 16, observation, tolerance) as problem:
            initial, optimization = _optimize(config, problem, directory)
            cancel_check("after_optimization")
            requested = pilot["initial_production_draws"]
            while True:
                samples, convergence = _converged_samples(
                    config,
                    problem,
                    initial,
                    directory / "posterior",
                    directory,
                    seed=seeds["sampling"],
                    spread=spread,
                    draws=requested,
                    cancel_check=cancel_check,
                )
                with samples:
                    cancel_check("before_rank_selection")
                    values = _saved_quantities(samples)
                    selection = _rank_selection(values)
                    if not selection["enough"]:
                        requested = max(
                            samples.metadata["total_draws"] * 2,
                            2 * selection["required_draws_per_chain"],
                        )
                        _event(
                            directory,
                            "extend_for_rank_independence",
                            replicate=replicate,
                            selection=selection,
                            new_total=requested,
                        )
                        continue
                    selected = (
                        values[
                            np.arange(4)[:, None],
                            np.asarray(selection["indices"]),
                        ]
                        .reshape(100, 5)
                        .T
                    )
                    doubled = (
                        values[
                            np.arange(4)[:, None],
                            np.asarray(selection["double_indices"]),
                        ]
                        .reshape(100, 5)
                        .T
                    )
                    ranks = sbc_ranks(truth, selected, seed=seeds["rank_ties"])
                    doubled_ranks = sbc_ranks(truth, doubled, seed=seeds["rank_ties"])
                    result = {
                        "replicate": replicate,
                        "status": "qualified",
                        "seeds": seeds,
                        "truth_quantities": truth,
                        "ranks": ranks,
                        "doubled_interval_ranks": doubled_ranks,
                        "selection": selection,
                        "optimization": optimization,
                        "convergence": asdict(convergence),
                        "problem_id": problem.problem_id,
                        "retained_per_chain": samples.metadata["total_draws"],
                    }
                    break
        result["checksums"] = _checksums(directory / "posterior")
        result["generation_sha256"] = generation["array_sha256"]
        _event(
            directory,
            "replicate_science_complete",
            replicate=index,
            ranks=result["ranks"],
            retained_per_chain=result["retained_per_chain"],
        )
        return _json_value(result)
    except (KeyboardInterrupt, InterruptedError) as exc:
        _event(directory, "interrupted", stage="sbc", replicate=index)
        if getattr(exc, "sbc_cooperative", False):
            exc.sbc_boundary = _validate_resume_artifacts(directory, seeds)
        raise
    except (HeatSolveFailure, NonfiniteProposal, _ScientificFailure) as exc:
        _failure(
            directory, directory / "failure.json", exc, replicate=index, seeds=seeds
        )
        return {
            "replicate": index,
            "status": "failed",
            "seeds": seeds,
            "failure": json.loads((directory / "failure.json").read_text()),
        }


def run_sbc(config, output, *, replicates=100, workers=1):
    r"""Run independent prior-predictive heat calibrations at nmesh=16.

    Every truth, noise vector and sampling calculation uses distinct recorded
    random streams. Optimization starts at the prior median, never the truth.
    Every completed posterior must meet ordinary four-chain qualification before
    100 balanced rank values are selected. The same retained chains extend when
    the doubled subsampling interval lacks enough draws. Failed replications are
    preserved, remain in the denominator, and prevent completion.

    ``workers`` bounds fresh replica processes, with one by default. Each replica
    retains its four-chain calculation. Worker count is execution provenance;
    fixed indices/seeds and all scientific settings remain in the study identity.

    Returns
    -------
    dict
        Individual statuses, ranks, sensitivity and finite-experiment uncertainty.
        Fewer than 100 requested replications can exercise mechanics only.
    """
    from hmc_sbc_worker import coordinate

    if not isinstance(config, StudyConfig):
        raise TypeError("config must be StudyConfig.")
    replicates = _integer(replicates, "replicates")
    workers = _integer(workers, "workers")
    if workers > replicates:
        raise ValueError("workers cannot exceed the fixed replica total.")
    pilot, pilot_hash = _qualified_pilot(config)
    output = Path(output).resolve()
    manifest = _study_manifest(
        config,
        output,
        kind="sbc",
        extra={"replicates": replicates, "pilot_qualification_sha256": pilot_hash},
    )
    _require_native(manifest)
    with _study_lock(output):
        records = coordinate(
            output,
            replicates=replicates,
            workers=workers,
            identity=manifest["identity_sha256"],
            validate_result=lambda directory, result: _validate_sbc_result(
                config, manifest, directory, result
            ),
            validate_resume=lambda directory: _validate_resume_artifacts(
                directory, _sbc_seeds(config, int(directory.name.split("-")[-1]))
            ),
            event=lambda kind, **evidence: _event(output, kind, **evidence),
            progress=lambda records: _write(
                output / "progress.json",
                {"requested": replicates, "records": records, "status": "incomplete"},
            ),
        )
        qualified = [record for record in records if record["status"] == "qualified"]
        summary = {
            "requested": replicates,
            "qualified": len(qualified),
            "records": records,
            "manifest_sha256": manifest["identity_sha256"],
            "calibration_complete": False,
            "status": "incomplete",
            "sampling_convergence": "qualified only for individually labelled replications",
            "numerical_correctness": "conditional on matching completed pilot qualification",
            "mesh_approximation": "not assessed",
            "sharp_truth_prior_adequacy": "not assessed",
            "rank_independence_assessment": "Worst quantile-indicator ESS determines subsampling; residual correlations and doubled-interval sensitivity remain explicitly reported.",
            "interpretation": "100 replications have limited sensitivity; no detected discrepancy is conditional evidence, not proof of exact sampling.",
        }
        if len(qualified) == replicates:
            ranks = np.asarray([r["ranks"] for r in qualified], dtype=int)
            doubled = np.asarray(
                [r["doubled_interval_ranks"] for r in qualified], dtype=int
            )
            envelope = sbc_uniform_envelope(
                replicates=replicates, posterior_count=100, quantities=5, seed=20261027
            )
            deviations = np.array(
                [_cdf_deviation(ranks[:, q], posterior_count=100) for q in range(5)]
            )
            double_deviations = np.array(
                [_cdf_deviation(doubled[:, q], posterior_count=100) for q in range(5)]
            )
            violations = (
                np.max(np.abs(deviations), axis=1) > envelope["critical_deviation"]
            )
            double_violations = (
                np.max(np.abs(double_deviations), axis=1)
                > envelope["critical_deviation"]
            )
            summary.update(
                envelope=envelope,
                cdf_deviations=deviations,
                doubled_cdf_deviations=double_deviations,
                violations=violations,
                doubled_interval_violations=double_violations,
                rank_change_on_doubling=doubled - ranks,
            )
            # Sensitivity is reported, not used as another unallocated formal test.
            investigate = bool(np.any(violations) or np.any(double_violations))
            summary["status"] = (
                "investigate_envelope_violation"
                if investigate
                else (
                    "conditional_calibration_evidence"
                    if replicates >= 100
                    else "mechanics_only"
                )
            )
            summary["calibration_complete"] = bool(
                replicates >= 100 and not investigate
            )
            _calibration_figures(output, ranks, doubled, envelope)
        _write(output / "calibration.json", summary)
        _event(
            output,
            "sbc_finished",
            status=summary["status"],
            requested=replicates,
            qualified=len(qualified),
            calibration_complete=summary["calibration_complete"],
        )
        return _json_value(summary)
