"""Shared fine-truth observations and physical mesh/prior comparisons.

Numerical accuracy, ordinary chain convergence, mesh approximation and prior
adequacy remain separate scientific questions. Imports of native libraries and
plotting are deferred until a physical calculation or figure is requested.
"""

from contextlib import ExitStack
from dataclasses import asdict
import json
import hashlib
import shutil
import subprocess
import sys
import math
from pathlib import Path
import time

import numpy as np

import hmc_validation as hv

TRUTHS = {
    "smooth": {"definition": "h=1+0.35*x+0.25*y+0.15*x*y", "seed": 20261026},
    "sharp": {"definition": "h=1 if x+0.25*y<0.62 else 3", "seed": 20261027},
}
PRIOR_CASES = {
    "baseline": (2.0, 1.0),
    "kappa-1": (1.0, 1.0),
    "kappa-4": (4.0, 1.0),
    "tau-0.5": (2.0, 0.5),
    "tau-2": (2.0, 2.0),
}


def _publication_paths(path):
    path = Path(path)
    return path.with_name(path.stem + "-sha256.json"), path.with_name(
        path.stem + "-pending.json"
    )


def _finish_publication(path, pending):
    digest_path, pending_path = _publication_paths(path)
    checksum = hashlib.sha256(hv._encoded(pending["payload"]) + b"\n").hexdigest()
    if checksum != pending["sha256"]:
        raise ValueError("Pending publication checksum changed.")
    hv._write(path, pending["payload"])
    hv._write(digest_path, {"sha256": checksum})
    pending_path.unlink()
    return pending["payload"]


def _publish_record(path, payload):
    """Durably stage both payload and checksum before publishing either file."""
    path = Path(path)
    _, pending_path = _publication_paths(path)
    pending = {
        "payload": hv._json_value(payload),
        "sha256": hashlib.sha256(hv._encoded(payload) + b"\n").hexdigest(),
    }
    if pending_path.exists():
        previous = _load_publication(path)
        if previous != pending["payload"]:
            raise ValueError(
                "Recover the preceding publication before replacing its evidence."
            )
        return previous
    hv._write(pending_path, pending)
    return _finish_publication(path, pending)


def _load_publication(path):
    path = Path(path)
    digest_path, pending_path = _publication_paths(path)
    if pending_path.exists():
        return _finish_publication(path, json.loads(pending_path.read_text()))
    if not digest_path.exists() or json.loads(digest_path.read_text())[
        "sha256"
    ] != hv._file_hash(path):
        raise ValueError("Saved publication checksum changed or is missing.")
    return json.loads(path.read_text())


def _verify_artifacts(output, result):
    for relative, checksum in result.get("checksums", {}).items():
        if hv._file_hash(output / relative) != checksum:
            raise ValueError("Saved mesh-study evidence changed.")


def _artifact_checksums(output):
    excluded = {
        "result.json",
        "result-sha256.json",
        "result-pending.json",
        "analysis-request.json",
        "analysis-request-sha256.json",
        "analysis-request-pending.json",
        "study.lock",
        "events.jsonl",
    }
    return {
        str(path.relative_to(output)): hv._file_hash(path)
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).parts[0] != "history"
        and path.name not in excluded
        and not path.name.endswith(("-wal", "-shm"))
    }


def _prepare_analysis(output, *, total_draws, reanalyze):
    """Record an explicit extension before altering any preserved trajectory."""
    output = Path(output)
    if total_draws is not None and reanalyze:
        raise ValueError("Choose either a larger retained total or reanalysis.")
    if total_draws is not None:
        hv._integer(total_draws, "total_draws")
    request_path = output / "analysis-request.json"
    result_path = output / "result.json"
    request_pending = _publication_paths(request_path)[1]
    if request_path.exists() or request_pending.exists():
        request = _load_publication(request_path)
        archive = output / request["archive"]
        previous = _load_publication(archive / "result.json")
        _verify_artifacts(archive, previous)
        if hv._file_hash(archive / "result.json") != request["previous_result_sha256"]:
            raise ValueError("Preserved extension request history changed.")
        completed = _load_publication(result_path)
        if completed.get("generation", 0) != request["generation"]:
            if (total_draws is not None and total_draws != request["total_draws"]) or (
                reanalyze and not request["reanalyze"]
            ):
                raise ValueError(
                    "An active analysis request cannot change during continuation."
                )
            return request, None
        _verify_artifacts(output, completed)
        request_path.unlink()
        _publication_paths(request_path)[0].unlink()
        if total_draws in (None, request["total_draws"]) and (
            not reanalyze or request["reanalyze"]
        ):
            return None, completed
    if not result_path.exists() and not _publication_paths(result_path)[1].exists():
        if reanalyze:
            raise ValueError("Reanalysis requires an existing sampled result.")
        return {"generation": 0, "total_draws": total_draws, "reanalyze": False}, None
    result = _load_publication(result_path)
    _verify_artifacts(output, result)
    if result["status"] == "failed":
        raise RuntimeError(
            "Preserved mesh-study failure requires investigation and a labelled new run."
        )
    if total_draws is None and not reanalyze:
        return None, result
    previous_total = result["requested_total"]
    if total_draws is not None and total_draws <= previous_total:
        raise ValueError(
            "Explicit extension requires a larger per-chain retained total."
        )
    generation = result.get("generation", 0)
    archive = output / "history" / f"generation-{generation:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    for relative, checksum in result.get("checksums", {}).items():
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and hv._file_hash(destination) != checksum:
            raise ValueError("Preserved generation artifact changed.")
        if not destination.exists():
            # A stopped copy can be overwritten on retry; only a verified copy
            # acquires a name in the preserved scientific generation.
            staged = archive.parent / f".{archive.name}-staging" / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / relative, staged)
            if hv._file_hash(staged) != checksum:
                raise ValueError("Staged generation artifact checksum changed.")
            staged.replace(destination)
    _publish_record(archive / "result.json", result)
    request = dict(
        generation=generation + 1,
        total_draws=total_draws or previous_total,
        reanalyze=total_draws is None,
        archive=str(archive.relative_to(output)),
        previous_result_sha256=hv._file_hash(archive / "result.json"),
    )
    _publish_record(request_path, request)
    return request, None


def _finish_analysis(output, result):
    _publish_record(output / "result.json", result)
    request_path = output / "analysis-request.json"
    if request_path.exists():
        request = _load_publication(request_path)
        if request["generation"] != result["generation"]:
            raise ValueError(
                "Completed result does not match the declared analysis generation."
            )
        request_path.unlink()
        _publication_paths(request_path)[0].unlink()


def _solve_totals(samples):
    committed, abandoned = {}, {}
    for chain in range(samples.metadata["chains"]):
        try:
            counters = samples._store.checkpoint(chain).counters
        except KeyError:
            continue
        for name, count in counters.items():
            committed[name] = committed.get(name, 0) + count
    failures = (
        list(samples._store.failures()) if hasattr(samples._store, "failures") else []
    )
    for failure in failures:
        for name, count in failure.get("aborted_work", {}).items():
            abandoned[name] = abandoned.get(name, 0) + count
    total = {
        name: committed.get(name, 0) + abandoned.get(name, 0)
        for name in committed.keys() | abandoned.keys()
    }
    return dict(committed=committed, abandoned=abandoned, total=total)


def _distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(
        count=int(values.size),
        nonfinite_count=int(values.size - finite.size),
        mean=float(finite.mean()) if finite.size else None,
        quantiles=(
            np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
            if finite.size
            else None
        ),
    )


def _partial_chain_evidence(samples, output):
    """Describe actual saved work even when chains cannot qualify or are uneven."""
    result = {
        "status": "available_unqualified_evidence",
        "counts": samples._counts().tolist(),
    }
    try:
        report = samples.diagnostics()
        result["diagnostics"] = asdict(report)
        minimum = hv._minimum_ess(report)
    except Exception as exc:
        result["diagnostics"] = None
        result["diagnostic_failure"] = {
            "exception": type(exc).__name__,
            "message": str(exc),
        }
        minimum = None
    totals = _solve_totals(samples)
    acceptance, energy = [], []
    retained_seconds = retained_solves = 0.0
    retained_count = 0
    correlations = {"acceptance": [], "energy_error": [], "physical": []}
    try:
        operator = hv._quantity_operator(samples._record)
    except Exception as exc:
        operator = None
        result["physical_autocorrelation_unresolved"] = str(exc)
    for chain in range(samples.metadata["chains"]):
        a, e, projections = [], [], []
        for row in samples._store.records(chain):
            a.append(row.statistics.get("acceptance_probability", np.nan))
            e.append(row.statistics.get("delta_energy", np.nan))
            if operator is not None:
                projections.append(row.h @ operator.T)
            retained_seconds += row.statistics.get("trajectory_seconds", np.nan)
            costs = row.statistics.get("solve_counts", {})
            retained_solves += costs.get("heat", np.nan) + costs.get("adjoint", np.nan)
            retained_count += 1
        acceptance.extend(a)
        energy.extend(e)
        lags = list(range(1, min(64, len(a) - 1) + 1))
        if projections:
            physical = np.asarray(projections)[None, :, :]
            correlations["physical"].append(
                {
                    "chain": chain,
                    "labels": ["area_mean"]
                    + [f"h({x},{y})" for x, y in hv.RANK_POINTS],
                    "lags": lags,
                    "values": [
                        hv._autocorrelation(physical, lag=k)[0].tolist() for k in lags
                    ],
                }
            )
        for name, values in (("acceptance", a), ("energy_error", e)):
            array = np.asarray(values, dtype=float)[None, :, None]
            correlations[name].append(
                {
                    "chain": chain,
                    "lags": lags,
                    "values": [
                        hv._autocorrelation(array, lag=k).ravel().tolist() for k in lags
                    ],
                }
            )
    total_solves = totals["total"].get("heat", 0) + totals["total"].get("adjoint", 0)
    wall = hv._sampling_seconds(output, samples.output)
    result.update(
        cost=dict(
            total_heat_adjoint_counts=totals["total"],
            committed_counts=totals["committed"],
            abandoned_counts=totals["abandoned"],
            sampling_wall_seconds=wall,
            retained_seconds=(
                float(retained_seconds)
                if retained_count and np.isfinite(retained_seconds)
                else None
            ),
            retained_heat_adjoint_solves=(
                float(retained_solves)
                if retained_count and np.isfinite(retained_solves)
                else None
            ),
        ),
        efficiency=dict(
            minimum_ess=minimum,
            ess_per_total_solve=(
                minimum / total_solves if minimum and total_solves else None
            ),
            ess_per_sampling_second=minimum / wall if minimum and wall > 0 else None,
        ),
        acceptance=_distribution(acceptance),
        energy_error=_distribution(energy),
        autocorrelation=correlations,
        failures=list(samples._store.failures()),
        interpretation="Available measurements from an unqualified or incomplete posterior; undefined efficiency remains unresolved.",
    )
    return hv._json_value(result)


def _failure_chain_evidence(output):
    if not (output / "posterior").exists():
        return {"status": "no_saved_posterior"}
    try:
        from posterior_samples import PosteriorSamples

        with PosteriorSamples.open(output / "posterior") as samples:
            return _partial_chain_evidence(samples, output)
    except Exception as exc:
        return {
            "status": "unresolved",
            "exception": type(exc).__name__,
            "message": str(exc),
        }


def _truth_values(truth, coordinates):
    x, y = np.asarray(coordinates).T
    if truth == "smooth":
        return 1 + 0.35 * x + 0.25 * y + 0.15 * x * y
    if truth == "sharp":
        return np.where(x + 0.25 * y < 0.62, 1.0, 3.0)
    raise ValueError("Unknown conductivity truth.")


def interface_coverage(coordinates, lower, upper, truth_values, *, width=0.05):
    r"""Report pointwise interval inclusion inside/outside a physical interface band.

    Parameters
    ----------
    coordinates : array-like, shape (N, 2)
        Physical nodal positions in the unit square.
    lower, upper, truth_values : array-like, shape (N,)
        Pointwise interval endpoints and the declared true conductivity.
    width : float
        Perpendicular distance from x + 0.25 y = 0.62 in unit-square lengths.

    Returns
    -------
    dict
        Counts and spatial fractions; an empty region has undefined coverage.
    """
    xy = np.asarray(coordinates, dtype=float)
    lo, hi, truth = (np.asarray(a, dtype=float) for a in (lower, upper, truth_values))
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or any(a.shape != (len(xy),) for a in (lo, hi, truth))
        or not all(np.isfinite(a).all() for a in (xy, lo, hi, truth))
        or np.any(lo > hi)
        or not np.isfinite(width)
        or width < 0
    ):
        raise ValueError(
            "Finite matching coordinates and ordered intervals are required."
        )
    near = np.abs(xy[:, 0] + 0.25 * xy[:, 1] - 0.62) / np.sqrt(1 + 0.25**2) <= width
    covered = (lo <= truth) & (truth <= hi)
    result = {
        "width": width,
        "interpretation": "spatial diagnostic from one field/noise realization; not repeated-noise coverage",
    }
    for name, mask in (("near", near), ("away", ~near)):
        result[name + "_count"] = int(mask.sum())
        result[name + "_coverage"] = float(covered[mask].mean()) if mask.any() else None
    return result


def _truth_temperatures(config, truth, nmesh):
    from benchmark_utils import _sharp_interface_h
    from forward_solver import SteadyHeat2DForwardSolver, _release_application_problem
    from temperature_observations import SensorOperator
    from heat_model import HeatNumerics

    coefficient = (
        _sharp_interface_h
        if truth == "sharp"
        else lambda x: 1 + 0.35 * x[0] + 0.25 * x[1] + 0.15 * x[0] * x[1]
    )
    forward = SteadyHeat2DForwardSolver(
        nmesh=nmesh,
        mesh_type=config.cell_type,
        h=coefficient,
        q=config.source,
        DBC_value=config.bottom_temperature,
        petsc_opts=HeatNumerics(
            rtol=1e-12, atol=1e-14, ksp_type="preonly", pc_type="lu"
        ).petsc_options(),
    )
    try:
        forward.solve()
        return SensorOperator(forward.V, config.sensor_points).predict(
            forward.T.x.array
        )
    finally:
        _release_application_problem(forward.problem)


def _physical_experiment(config):
    return {
        key: getattr(config, key)
        for key in (
            "cell_type",
            "source",
            "bottom_temperature",
            "sigma",
            "conductivity_units",
        )
    }


def make_truth_data(config, *, truth, output):
    r"""Freeze one fine-mesh observation realization after sensor refinement.

    Parameters
    ----------
    config : StudyConfig
        Declared heat source, boundaries, sensor noise and numerical identity.
    truth : {'smooth', 'sharp'}
        Predeclared physical conductivity and noise seed.
    output : path-like
        Dedicated immutable observation directory; a matching call reuses it.

    Returns
    -------
    pathlib.Path
        NPZ with exact sensor order, values, sigma and noiseless temperatures.
    """
    if truth not in TRUTHS:
        raise ValueError("truth must be smooth or sharp.")
    output = Path(output).resolve()
    manifest = hv._study_manifest(
        config, output, kind="fine_truth", extra={"truth": TRUTHS[truth], "case": truth}
    )
    hv._require_native(manifest)
    path = output / "observations.npz"
    with hv._study_lock(output):
        if path.exists() or path.with_suffix(".unpublished.json").exists():
            _truth_observations(config, path, _locked=True)
            return path
        if (output / "failure.json").exists():
            raise RuntimeError(
                "Preserved truth refinement failure requires investigation."
            )
        try:
            refinement = []
            nmesh = 128
            coarse = _truth_temperatures(config, truth, nmesh)
            while True:
                fine = _truth_temperatures(config, truth, 2 * nmesh)
                if (
                    coarse.shape != (256,)
                    or fine.shape != (256,)
                    or not np.isfinite([coarse, fine]).all()
                ):
                    raise ValueError(
                        "Truth temperatures must be finite at every sensor."
                    )
                change = float(np.max(np.abs(fine - coarse) / config.sigma))
                refinement.append(
                    {
                        "coarse": nmesh,
                        "fine": 2 * nmesh,
                        "maximum_change_over_sigma": change,
                    }
                )
                hv._write(output / "refinement.json", refinement)
                hv._event(output, "truth_refinement", **refinement[-1])
                if change < 0.1:
                    break
                nmesh *= 2
                coarse = fine
            values = fine + np.random.default_rng(TRUTHS[truth]["seed"]).normal(
                0, config.sigma, 256
            )
            record = hv._save_arrays(
                path,
                path.with_suffix(".unpublished.json"),
                dict(
                    points=config.sensor_points,
                    values=values,
                    sigma=np.asarray(config.sigma),
                    noiseless=fine,
                ),
                dict(
                    case=truth,
                    **TRUTHS[truth],
                    truth_mesh=2 * nmesh,
                    refinement=refinement,
                    refinement_sha256=hv._file_hash(output / "refinement.json"),
                    experiment=_physical_experiment(config),
                    manifest_sha256=manifest["identity_sha256"],
                    status="qualified_reference",
                    truth_representation="continuous Q1 nodal interpolant, including transition cells for sharp truth",
                ),
            )
            _publish_record(path.with_suffix(".json"), record)
            path.with_suffix(".unpublished.json").unlink()
            return path
        except (KeyboardInterrupt, InterruptedError):
            hv._event(output, "interrupted", stage="truth_refinement")
            raise
        except Exception as exc:
            hv._failure(output, output / "failure.json", exc, stage="truth_refinement")
            raise


def _truth_observations(config, path, *, _locked=False):
    path = Path(path).resolve()
    if not _locked:
        with hv._study_lock(path.parent):
            return _truth_observations(config, path, _locked=True)
    record_path = path.with_suffix(".json")
    unpublished = path.with_suffix(".unpublished.json")
    if (
        not record_path.exists()
        and unpublished.exists()
        and not _publication_paths(record_path)[1].exists()
    ):
        record = hv._array_record(path, unpublished)
        _publish_record(record_path, record)
    record = _load_publication(record_path)
    hv._array_record(path, record_path)
    if hv._file_hash(path.parent / "refinement.json") != record["refinement_sha256"]:
        raise ValueError("Fine-truth refinement archive checksum changed.")
    if (
        record["status"] != "qualified_reference"
        or record["case"] not in TRUTHS
        or record["experiment"] != _physical_experiment(config)
        or record["seed"] != TRUTHS[record["case"]]["seed"]
        or record["definition"] != TRUTHS[record["case"]]["definition"]
        or not record["refinement"]
        or record["refinement"][-1]["maximum_change_over_sigma"] >= 0.1
    ):
        raise ValueError("Truth record does not qualify this physical experiment.")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (
        manifest.get("identity_sha256") != hv._hash(manifest["identity"])
        or record["manifest_sha256"] != manifest["identity_sha256"]
        or manifest["identity"]["kind"] != "fine_truth"
        or manifest["identity"]["extra"]["case"] != record["case"]
    ):
        raise ValueError("Fine-truth manifest identity changed.")
    expected_coarse = 128
    for comparison in record["refinement"]:
        change = comparison["maximum_change_over_sigma"]
        if (
            comparison["coarse"] != expected_coarse
            or comparison["fine"] != 2 * expected_coarse
            or not np.isfinite(change)
            or change < 0
        ):
            raise ValueError("Fine-truth refinement sequence changed.")
        expected_coarse *= 2
    if record["truth_mesh"] != expected_coarse:
        raise ValueError("Truth mesh does not match its refinement evidence.")
    return hv._observations(config, path), record


def _prior_case(config, nmesh):
    if nmesh not in (16, 32, 64) or config.mu != math.log(2.0) or config.h_ref != 1.0:
        raise ValueError("Mesh study requires declared physical baseline mu and h_ref.")
    for name, pair in PRIOR_CASES.items():
        if (config.kappa, config.tau) == pair and (name == "baseline" or nmesh == 16):
            return name
    raise ValueError("Only predeclared one-at-a-time n16 prior sensitivity is allowed.")


def _ess(values):
    from posterior_diagnostics import _load_backend

    backend, _ = _load_backend()
    kw = dict(chain_axis=0, draw_axis=1)
    return np.minimum(
        backend.ess(values, method="bulk", **kw),
        backend.ess(values, method="tail", prob=(0.05, 0.95), **kw),
    )


def _block_uncertainty(
    values, acceptance, seconds, solves, *, correlation_time, seed, ess=None
):
    """Paired within-chain block resampling; no chain is replaced by another."""
    ess = _ess if ess is None else ess
    values, acceptance, seconds, solves = map(
        np.asarray, (values, acceptance, seconds, solves)
    )
    result = {
        "status": "unresolved",
        "resamples": 500,
        "seed": seed,
        "interval_probabilities": [0.025, 0.975],
        "minimum_blocks_per_chain": 8,
    }
    if (
        values.ndim != 3
        or values.shape[0] != 4
        or any(a.shape != values.shape[:2] for a in (acceptance, seconds, solves))
        or not all(np.isfinite(a).all() for a in (values, acceptance, seconds, solves))
        or not np.isfinite(correlation_time)
        or correlation_time <= 0
        or np.any(seconds <= 0)
        or np.any(solves <= 0)
    ):
        return {
            **result,
            "reason": "Missing/nonfinite paired trajectory cost or correlation evidence.",
        }
    length = max(2, int(math.ceil(2 * correlation_time)))
    count = values.shape[1] // length
    result.update(
        block_length=length,
        blocks_per_chain=count,
        discarded_tail_per_chain=values.shape[1] - count * length,
    )
    if count < 8:
        return {**result, "reason": "Too few contiguous blocks per chain."}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(500):
        index = rng.integers(0, count, size=(4, count))[..., None] * length + np.arange(
            length
        )
        index = index.reshape(4, -1)
        chain = np.arange(4)[:, None]
        effective = np.asarray(ess(values[chain, index]), dtype=float)
        estimates.append(
            np.r_[
                np.mean(acceptance[chain, index]),
                effective / seconds[chain, index].sum(),
                effective / solves[chain, index].sum(),
            ]
        )
    estimates = np.asarray(estimates)
    if not np.isfinite(estimates).all() or np.any(estimates[:, 1:] <= 0):
        return {**result, "reason": "Resampled ESS is missing or nonfinite."}
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0).T
    size = values.shape[2]
    return {
        **result,
        "status": "resolved",
        "acceptance_interval": intervals[0].tolist(),
        "ess_per_second_intervals": intervals[1 : 1 + size].tolist(),
        "ess_per_solve_intervals": intervals[1 + size :].tolist(),
        "acceptance_resamples": estimates[:, 0].tolist(),
        "ess_per_second_resamples": estimates[:, 1 : 1 + size].tolist(),
        "ess_per_solve_resamples": estimates[:, 1 + size :].tolist(),
        "interpretation": "Monte Carlo uncertainty conditional on the retained posterior and measured trajectory costs",
    }


def _predictive_summary(predictions, observed, sigma, *, seed):
    predictions = np.asarray(predictions)
    mean = predictions.mean(axis=(0, 1))
    sd = np.sqrt(predictions.var(axis=(0, 1)) + np.asarray(sigma) ** 2)
    noisy = (
        predictions + np.random.default_rng(seed).normal(size=predictions.shape) * sigma
    )
    lo, hi = np.quantile(noisy, [0.05, 0.95], axis=(0, 1))
    return hv._json_value(
        dict(
            mean=mean,
            predictive_sd=sd,
            lower=lo,
            upper=hi,
            standardized_residual=(mean - observed) / sd,
            observed=observed,
            latent_sd=predictions.std(axis=(0, 1)),
            noise_seed=seed,
            interpretation="posterior predictive temperatures include independent measurement noise",
        )
    )


def _correlation_time(values):
    """Conservative maximum initial-positive-sequence time over chains/columns."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    largest = 1.0
    for chain in values:
        for x in chain.T:
            if not np.isfinite(x).all():
                return float("nan")
            x = x - x.mean()
            if np.all(x == 0):
                continue
            n = len(x)
            transform = np.fft.rfft(x, n=2 * n)
            covariance = np.fft.irfft(transform * transform.conj(), n=2 * n)[:n]
            correlation = covariance / covariance[0]
            pairs = correlation[1 : 1 + 2 * ((n - 1) // 2)].reshape(-1, 2).sum(axis=1)
            positive = pairs[
                : next((i for i, x in enumerate(pairs) if x <= 0), len(pairs))
            ]
            if len(positive):
                positive = np.minimum.accumulate(positive)
            largest = max(largest, 1 + 2 * positive.sum())
    return float(largest)


def _physical_evidence(config, samples, report, truth, output):
    summary = samples.summary()
    n = int(samples._counts()[0])
    quantities = hv._saved_quantities(samples)
    conductivity = report.per_quantity["conductivity"]
    nodal_ess = np.minimum(conductivity["bulk_ess"], conductivity["tail_ess"])
    worst = int(np.argmin(nodal_ess))
    worst_group, worst_index, worst_score = None, None, float("inf")
    for group, diagnostic in report.per_quantity.items():
        scores = np.minimum(diagnostic["bulk_ess"], diagnostic["tail_ess"])
        scores = np.where(diagnostic["analytically_fixed"], np.inf, scores)
        index = int(np.argmin(scores))
        if scores[index] < worst_score:
            worst_group, worst_index, worst_score = group, index, float(scores[index])
    from posterior_diagnostics import _physical_observables

    physical_weights, _ = _physical_observables(samples._record)
    values = np.empty((4, n, 7))
    values[..., :5] = quantities
    acceptance, seconds, solves, energies = (np.empty((4, n)) for _ in range(4))
    solve_totals = _solve_totals(samples)
    total_counts = solve_totals["total"]
    for chain in range(4):
        for draw, row in enumerate(samples._store.records(chain)):
            values[chain, draw, 5] = row.h[worst]
            if worst_group == "conductivity":
                values[chain, draw, 6] = row.h[worst_index]
            elif worst_group == "sensor_temperature":
                values[chain, draw, 6] = row.sensor_temperature[worst_index]
            else:
                weight_index = (
                    "area_mean",
                    "left_minus_right",
                    "bottom_minus_top",
                ).index(worst_group)
                values[chain, draw, 6] = row.h @ physical_weights[weight_index]
            stats = row.statistics
            acceptance[chain, draw] = stats["acceptance_probability"]
            energies[chain, draw] = (
                stats["delta_energy"] if stats["delta_energy"] is not None else np.nan
            )
            seconds[chain, draw] = stats.get("trajectory_seconds", np.nan)
            costs = stats.get("solve_counts", {})
            solves[chain, draw] = costs.get("heat", np.nan) + costs.get(
                "adjoint", np.nan
            )
    all_ess = [
        np.asarray(item[key])[~np.asarray(item["analytically_fixed"], dtype=bool)]
        for item in report.per_quantity.values()
        for key in ("bulk_ess", "tail_ess")
    ]
    correlation = max(
        1.0,
        4 * n / min(float(x.min()) for x in all_ess),
        _correlation_time(acceptance),
        _correlation_time(energies),
        _correlation_time(values),
        _correlation_time(seconds),
        _correlation_time(solves),
    )
    uncertainty = _block_uncertainty(
        values,
        acceptance,
        seconds,
        solves,
        correlation_time=correlation,
        seed=hv._seed(
            config,
            140,
            len(samples.coordinates),
            TRUTHS[truth]["seed"],
            list(PRIOR_CASES.values()).index((config.kappa, config.tau)),
        ),
    )
    point_nodes = [
        int(
            np.flatnonzero(
                np.all(
                    np.isclose(samples.coordinates, point, atol=1e-13, rtol=0), axis=1
                )
            )[0]
        )
        for point in hv.RANK_POINTS
    ]
    labels = (
        ["area_mean"]
        + [f"h({x},{y})" for x, y in hv.RANK_POINTS]
        + ["worst_nodal_ess", f"worst_monitored_ess:{worst_group}:{worst_index}"]
    )
    means = values.mean(axis=(0, 1))
    intervals = np.quantile(values, [0.05, 0.5, 0.95], axis=(0, 1))
    shared = []
    for i, label in enumerate(labels):
        diagnostic = report.per_quantity["area_mean"] if i == 0 else conductivity
        index = 0 if i == 0 else (point_nodes[i - 1] if i < 5 else worst)
        location = None if i == 0 else samples.coordinates[index].tolist()
        if i == 6:
            diagnostic = report.per_quantity[worst_group]
            index = worst_index
            location = diagnostic["locations"][index]
        shared.append(
            dict(
                label=label,
                location=location,
                mean=float(means[i]),
                quantiles=intervals[:, i].tolist(),
                mean_mcse=diagnostic["mean_mcse"][index],
                quantile_mcse=np.asarray(diagnostic["quantile_mcse"])[
                    :, index
                ].tolist(),
                bulk_ess=diagnostic["bulk_ess"][index],
                tail_ess=diagnostic["tail_ess"][index],
            )
        )
    predictions = {
        name: []
        for name in (
            "mean",
            "predictive_sd",
            "lower",
            "upper",
            "standardized_residual",
            "observed",
            "latent_sd",
        )
    }
    observation = samples._record.arrays
    # Bound the sensor axis in memory; never load the full conductivity chains.
    for start in range(0, len(observation["sensor_values"]), 32):
        stop = min(start + 32, len(observation["sensor_values"]))
        block = np.empty((4, n, stop - start))
        for chain in range(4):
            for draw, row in enumerate(samples._store.records(chain)):
                block[chain, draw] = row.sensor_temperature[start:stop]
        sigma = np.asarray(observation["sensor_sigma"])
        part = _predictive_summary(
            block,
            observation["sensor_values"][start:stop],
            sigma if sigma.ndim == 0 else sigma[start:stop],
            seed=hv._seed(config, 141, start),
        )
        for name in predictions:
            predictions[name].extend(part[name])
    truth_values = _truth_values(truth, summary.coordinates)
    hv._save_arrays(
        output / "physical-summary.npz",
        output / "physical-summary.json",
        dict(
            coordinates=summary.coordinates,
            cells=summary.cells,
            truth=truth_values,
            mean=summary.mean,
            lower=summary.quantiles[0],
            median=summary.quantiles[1],
            upper=summary.quantiles[2],
        ),
        dict(
            probabilities=[0.05, 0.5, 0.95],
            interpretation="physical conductivity after transformation",
        ),
    )
    effective = _ess(values)
    retained_seconds = float(seconds.sum()) if np.isfinite(seconds).all() else None
    retained_solves = float(solves.sum()) if np.isfinite(solves).all() else None
    return hv._json_value(
        dict(
            shared_locations=shared,
            all_node_diagnostics=asdict(report),
            coverage=interface_coverage(
                summary.coordinates,
                summary.quantiles[0],
                summary.quantiles[2],
                truth_values,
            ),
            predictive={
                **predictions,
                "points": observation["sensor_points"],
                "includes_observation_noise": True,
                "noise_seed_rule": "SeedSequence([config.seed,141,sensor_block_start])",
            },
            cost=dict(
                total_heat_adjoint_counts=total_counts,
                committed_counts=solve_totals["committed"],
                abandoned_counts=solve_totals["abandoned"],
                retained_seconds=retained_seconds,
                retained_heat_adjoint_solves=retained_solves,
                sampling_wall_seconds=hv._sampling_seconds(output, samples.output),
                labels=labels,
                ess=effective,
                ess_per_second=(
                    effective / retained_seconds if retained_seconds else None
                ),
                ess_per_solve=effective / retained_solves if retained_solves else None,
                ess_per_sampling_wall_second=(
                    effective / hv._sampling_seconds(output, samples.output)
                    if hv._sampling_seconds(output, samples.output) > 0
                    else None
                ),
                ess_per_total_heat_adjoint_solve=(
                    effective / (total_counts["heat"] + total_counts["adjoint"])
                    if total_counts["heat"] + total_counts["adjoint"] > 0
                    else None
                ),
                interpretation="retained seconds sum measured trajectory wall times; sampling wall includes warmup, resume and storage overhead",
            ),
            uncertainty={
                **uncertainty,
                "quantity_labels": labels,
                "largest_measured_correlation_time": correlation,
            },
            acceptance=dict(
                mean=acceptance.mean(),
                quantiles=np.quantile(acceptance, [0.01, 0.05, 0.5, 0.95, 0.99]),
            ),
            energy_error=dict(
                quantiles=np.quantile(energies, [0.01, 0.05, 0.5, 0.95, 0.99]),
                nonfinite_count=int((~np.isfinite(energies)).sum()),
            ),
            autocorrelation={
                "quantity_labels": labels,
                "lags": list(range(1, min(64, n - 1) + 1)),
                "values": [
                    hv._autocorrelation(values, lag=k)
                    for k in range(1, min(64, n - 1) + 1)
                ],
            },
        )
    )


def _study_figures(output, evidence):
    """Render outside the numerical interpreter to preserve its native identity."""
    output = Path(output).resolve()
    figure_input = output / "figure-input.json"
    hv._write(figure_input, {"predictive": evidence["predictive"]})
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--render-figures",
            str(output),
            str(figure_input),
        ],
        check=True,
    )


def _render_study_figures(output, evidence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(output / "physical-summary.npz", allow_pickle=False) as data:
        xy = data["coordinates"]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for axis, name, values in zip(
            axes,
            ("Truth", "Posterior mean", "90% interval width"),
            (data["truth"], data["mean"], data["upper"] - data["lower"]),
        ):
            limits = (
                {}
                if name == "90% interval width"
                else {
                    "vmin": min(float(data["truth"].min()), float(data["mean"].min())),
                    "vmax": max(float(data["truth"].max()), float(data["mean"].max())),
                }
            )
            artist = axis.scatter(*xy.T, c=values, s=8, rasterized=True, **limits)
            axis.set(xlabel="x", ylabel="y", title=name, aspect="equal")
            figure.colorbar(artist, ax=axis)
        figure.savefig(output / "conductivity.png", dpi=160)
        plt.close(figure)
    predictive = evidence["predictive"]
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    artist = axis.scatter(
        *np.asarray(predictive["points"]).T,
        c=predictive["standardized_residual"],
        cmap="coolwarm",
    )
    axis.set(
        xlabel="x",
        ylabel="y",
        title="Predictive residual / predictive SD",
        aspect="equal",
    )
    figure.colorbar(artist, ax=axis)
    figure.savefig(output / "predictive-residuals.png", dpi=160)
    plt.close(figure)


def _total_wall_seconds(output, started):
    path = output / "events.jsonl"
    previous = (
        sum(
            item["seconds"]
            for item in map(json.loads, path.read_text().splitlines())
            if item["kind"] == "mesh_invocation_elapsed"
        )
        if path.exists()
        else 0.0
    )
    return previous + time.monotonic() - started


def run_mesh_study(
    config, *, data_path, nmesh, output, total_draws=None, reanalyze=False
):
    r"""Sample a declared conductivity posterior using frozen fine-truth data.

    Parameters
    ----------
    config : StudyConfig
        Physical prior and path to a completed data-specific numerical pilot.
    data_path : path-like
        Qualified reference NPZ created by :func:`make_truth_data`.
    nmesh : {16, 32, 64}
        Inference resolution; prior sensitivity cases use 16 only.
    output : path-like
        Dedicated calculation directory, preserving failures and continuations.
    total_draws : int, optional
        Explicit larger retained total per chain; prior generations are archived.
    reanalyze : bool, optional
        Recompute summaries and figures of existing draws without extending them.

    Returns
    -------
    dict
        Actual convergence, physical summaries, costs and unresolved criteria.
    """
    prior_case = _prior_case(config, nmesh)
    observations, truth = _truth_observations(config, data_path)
    output = Path(output).resolve()
    manifest = hv._study_manifest(
        config,
        output,
        kind="mesh",
        extra=dict(
            nmesh=nmesh,
            truth=truth["case"],
            prior_case=prior_case,
            truth_array_sha256=truth["array_sha256"],
            truth_record_sha256=hv._file_hash(Path(data_path).with_suffix(".json")),
            observation_sha256=hv._hash([a.tolist() for a in observations]),
        ),
    )
    hv._require_native(manifest)
    with hv._study_lock(output):
        result_path = output / "result.json"
        request, cached = _prepare_analysis(
            output, total_draws=total_draws, reanalyze=reanalyze
        )
        if cached is not None:
            return cached
        started = time.monotonic()
        base = dict(
            manifest_sha256=manifest["identity_sha256"],
            generation=request["generation"],
            analysis_request=request,
            truth=truth["case"],
            nmesh=nmesh,
            prior_case=prior_case,
            truth_array_sha256=truth["array_sha256"],
            physical_prior={
                key: getattr(config, key) for key in ("mu", "kappa", "tau", "h_ref")
            },
        )
        try:
            pilot, pilot_hash = hv._qualified_pilot(config)
            pilot_manifest = json.loads(
                (Path(config.pilot_output) / "manifest.json").read_text()
            )
            if (
                pilot_manifest["identity"]["extra"]["observation_sha256"]
                != manifest["identity"]["extra"]["observation_sha256"]
            ):
                raise ValueError(
                    "Numerical pilot observations do not match this actual truth data."
                )
            pilot_observations = hv._observations(
                config, Path(config.pilot_output) / "observations.npz"
            )
            if any(
                a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
                for a, b in zip(observations, pilot_observations)
            ):
                raise ValueError(
                    "Pilot observation bytes do not match the frozen truth data."
                )
            hv._write(output / "pilot-qualification.json", pilot)
            hv._write(output / "pilot-manifest.json", pilot_manifest)
            hv._write(output / "truth-reference.json", truth)
            mesh = pilot["meshes"][str(nmesh)]
            tolerance = tuple(mesh["tolerances"])
            with hv._problem(config, nmesh, observations, tolerance) as problem:
                initial, optimization = hv._optimize(config, problem, output)
                if not optimization["success"]:
                    raise hv._ScientificFailure(
                        "Log-conductivity mode optimization did not converge."
                    )
                if request["reanalyze"]:
                    from posterior_samples import PosteriorSamples

                    samples = PosteriorSamples.open(output / "posterior")
                    report = samples.diagnostics()
                    if not report.qualified:
                        samples.close()
                        raise hv._ScientificFailure(
                            "Reanalysis did not qualify the saved posterior."
                        )
                else:
                    samples, report = hv._converged_samples(
                        config,
                        problem,
                        initial,
                        output / "posterior",
                        output,
                        seed=hv._seed(
                            config,
                            142,
                            nmesh,
                            TRUTHS[truth["case"]]["seed"],
                            list(PRIOR_CASES).index(prior_case),
                        ),
                        spread=mesh["spread"],
                        draws=max(
                            pilot["initial_production_draws"],
                            request["total_draws"] or 0,
                        ),
                    )
                try:
                    requested_total = samples.metadata["total_draws"]
                    evidence = _physical_evidence(
                        config, samples, report, truth["case"], output
                    )
                    chain_evidence = hv._chain_evidence(
                        samples.output, config.step_size_bounds[0]
                    )
                    # The pilot's 256/256 spread screen is not a production criterion.
                    chain_evidence.pop("passes_spread_criteria", None)
                    for chain in chain_evidence["chains"]:
                        chain.pop("passes_spread_criteria", None)
                finally:
                    samples.close()
            _study_figures(output, evidence)
            uncertainty_resolved = evidence["uncertainty"]["status"] == "resolved"
            result = dict(
                **base,
                status="sampled",
                requested_total=requested_total,
                convergence_qualified=report.qualified,
                cost_uncertainty_resolved=uncertainty_resolved,
                numerical_pilot_sha256=pilot_hash,
                tolerances=tolerance,
                truth_reference=truth,
                optimization=optimization,
                chain_evidence=chain_evidence,
                **evidence,
                total_wall_seconds=_total_wall_seconds(output, started),
                open_requirements=(
                    []
                    if uncertainty_resolved
                    else [
                        "Cost uncertainty unresolved: more independent contiguous blocks or missing cost measurements."
                    ]
                ),
                interpretation="Convergence concerns this discrete posterior; mesh sensitivity and prior adequacy are separate conclusions.",
            )
            result["checksums"] = _artifact_checksums(output)
            _finish_analysis(output, result)
            return hv._json_value(result)
        except (KeyboardInterrupt, InterruptedError):
            hv._event(output, "interrupted", stage="mesh")
            raise
        except Exception as exc:
            hv._failure(output, output / "failure.json", exc, stage="mesh")
            result = dict(
                **base,
                status="failed",
                available_evidence=_failure_chain_evidence(output),
                convergence_qualified=False,
                failure={"exception": type(exc).__name__, "message": str(exc)},
                total_wall_seconds=_total_wall_seconds(output, started),
                convergence=(
                    json.loads((output / "convergence.json").read_text())
                    if (output / "convergence.json").exists()
                    else None
                ),
                sampling_wall_seconds=hv._sampling_seconds(
                    output, output / "posterior"
                ),
                preserved_artifacts=[
                    str(path.relative_to(output))
                    for path in output.rglob("*")
                    if path.is_file()
                ],
                checksums={"failure.json": hv._file_hash(output / "failure.json")},
            )
            result["checksums"] = _artifact_checksums(output)
            _finish_analysis(output, result)
            return result
        finally:
            hv._event(
                output, "mesh_invocation_elapsed", seconds=time.monotonic() - started
            )


def write_study_report(manifests, destination):
    r"""Write an evidence-linked comparison, keeping missing/failed cases visible.

    Parameters
    ----------
    manifests : iterable of path-like
        Mesh-study manifest paths; unsuccessful and repeated attempts are retained.
    destination : path-like
        Markdown report path. No missing calculation is inferred or substituted.

    Returns
    -------
    None
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    identities = {truth: set() for truth in TRUTHS}
    # Recovery writes result/request files. Hold each physical calculation's
    # ownership lock through history discovery, recovery and artifact checks.
    with ExitStack() as ownership:
        owners = set()
        candidates = []
        for item in manifests:
            path = Path(item).resolve()
            if path.is_dir():
                path /= "manifest.json"
            owner = path.parent
            if owner.parent.name == "history" and owner.name.startswith("generation-"):
                owner = owner.parent.parent
            if owner not in owners:
                ownership.enter_context(hv._study_lock(owner))
                owners.add(owner)
            candidates.extend(
                sorted((path.parent / "history").glob("generation-*/manifest.json"))
            )
            candidates.append(path)
        for path in dict.fromkeys(candidates):
            manifest = json.loads(path.read_text())
            if (
                manifest.get("identity_sha256") != hv._hash(manifest["identity"])
                or manifest["identity"]["kind"] != "mesh"
            ):
                raise ValueError("Report requires intact mesh-study manifests.")
            identity = manifest["identity"]["extra"]
            result_path = path.parent / "result.json"
            result = (
                _load_publication(result_path)
                if result_path.exists() or _publication_paths(result_path)[1].exists()
                else {
                    "status": "incomplete",
                    "truth": identity["truth"],
                    "nmesh": identity["nmesh"],
                    "prior_case": identity["prior_case"],
                    "manifest_sha256": manifest["identity_sha256"],
                }
            )
            if result["manifest_sha256"] != manifest["identity_sha256"]:
                raise ValueError("Result belongs to another physical calculation.")
            request_path = path.parent / "analysis-request.json"
            if request_path.exists() or _publication_paths(request_path)[1].exists():
                request = _load_publication(request_path)
                if result.get("generation", 0) != request["generation"]:
                    result = dict(
                        status="extension_in_progress",
                        truth=identity["truth"],
                        nmesh=identity["nmesh"],
                        prior_case=identity["prior_case"],
                        manifest_sha256=manifest["identity_sha256"],
                        generation=request["generation"],
                        analysis_request=request,
                    )
            _verify_artifacts(path.parent, result)
            identities[identity["truth"]].add(identity["truth_array_sha256"])
            records.append((path, result))
    if any(len(values) > 1 for values in identities.values()):
        raise ValueError(
            "All meshes and prior sensitivity cases must share byte-identical truth data."
        )
    lines = [
        "# Conductivity mesh and prior comparison",
        "",
        "This report separates numerical accuracy, chain convergence, mesh approximation and prior adequacy.",
        "The scientific study is not complete while any required outcome or uncertainty is unresolved; Task 13 same-model calibration is independent evidence.",
        "",
        "| Truth | nmesh | Prior case | Outcome | Converged | Cost uncertainty | Manifest |",
        "|---|---:|---|---|---|---|---|",
    ]
    required = [
        (truth, nmesh, "baseline") for truth in TRUTHS for nmesh in (16, 32, 64)
    ]
    required += [
        (truth, 16, case)
        for truth in TRUTHS
        for case in PRIOR_CASES
        if case != "baseline"
    ]
    for truth, nmesh, case in required:
        matching = [
            (path, result)
            for path, result in records
            if (result["truth"], result["nmesh"], result["prior_case"])
            == (truth, nmesh, case)
        ]
        if not matching:
            lines.append(
                f"| {truth} | {nmesh} | {case} | missing | unresolved | unresolved | — |"
            )
        for path, result in matching:
            lines.append(
                f"| {truth} | {nmesh} | {case} | {result['status']} | {result.get('convergence_qualified', False)} | {result.get('cost_uncertainty_resolved', False)} | [{path.parent.name}]({path}) |"
            )
    lines += [
        "",
        "The Gaussian prior is held at mu=log(2), kappa=2, tau=1, h_ref=1 across baseline meshes. Each n16 sensitivity changes only the indicated kappa or tau and represents a separate posterior; these cases cannot be selected to force truth recovery.",
        "",
        "Interface inclusion is a spatial diagnostic from one field/noise realization. The near region is abs(x+0.25*y-0.62)/sqrt(1+0.25**2)<=0.05. Nominal 90% spatial inclusion and perfect interface recovery are not correctness criteria. Repeated-noise coverage requires a separate repeated experiment. Continuous nodal conductivity represents a jump through transition cells; an exactly discontinuous model may require a different field representation and prior.",
        "",
        "Mean and interval-endpoint Monte Carlo errors describe finite sampling of each fixed posterior. Differences between mesh or prior cases describe different posterior laws. Shared observations do not make those differences Monte Carlo error. Predictive residuals include measurement noise.",
    ]
    for path, result in records:
        title = f"{result['truth']}, nmesh={result['nmesh']}, {result['prior_case']}: {path.parent.name}"
        lines += [
            "",
            f"## {title}",
            "",
            f"Manifest identity: `{result['manifest_sha256']}`. Manifest file SHA256: `{hv._file_hash(path)}`.",
            f"[Full recorded evidence]({path.parent / 'result.json'}). Generation {result.get('generation', 0)}.",
        ]
        if result["status"] != "sampled":
            lines += [
                "",
                f"Unresolved outcome: {result.get('failure', result['status'])}.",
            ]
            partial = result.get("available_evidence", {})
            lines += [
                "",
                f"Available saved-chain evidence: {partial.get('status', 'unavailable')}.",
            ]
            for name in (
                "cost",
                "efficiency",
                "acceptance",
                "energy_error",
                "autocorrelation",
                "diagnostics",
                "diagnostic_failure",
            ):
                if name in partial:
                    lines += [
                        "",
                        f"{name}: `{json.dumps(partial[name], sort_keys=True)}`.",
                    ]
            continue
        lines += [
            "",
            f"Numerical pilot SHA256: `{result['numerical_pilot_sha256']}`. Fixed solve tolerances: {result['tolerances']}.",
            f"Truth array SHA256: `{result['truth_array_sha256']}`.",
            "",
            f"Interface-region inclusion: {result['coverage']}.",
            "",
            f"Acceptance: {result['acceptance']}. Hamiltonian energy errors: {result['energy_error']}.",
            "",
            f"Measured costs and efficiencies: `{json.dumps(result['cost'], sort_keys=True)}`.",
            "",
            f"500 seeded contiguous-block uncertainty: `{json.dumps({k:v for k,v in result['uncertainty'].items() if not k.endswith('_resamples')}, sort_keys=True)}`.",
            "",
            f"All-node worst diagnostics and physical locations: `{json.dumps(result['all_node_diagnostics']['worst_locations'], sort_keys=True)}`.",
            "",
            "| Physical quantity | Mean | Mean MCSE | 5%, 50%, 95% quantiles | Quantile MCSE |",
            "|---|---:|---:|---|---|",
        ]
        for quantity in result["shared_locations"]:
            lines.append(
                f"| {quantity['label']} | {quantity['mean']:.8g} | {quantity['mean_mcse']:.4g} | {quantity['quantiles']} | {quantity['quantile_mcse']} |"
            )
        lines += [
            "",
            f"![Truth, posterior mean, and interval width]({path.parent / 'conductivity.png'})",
            "",
            f"![Fixed-sensor predictive residuals]({path.parent / 'predictive-residuals.png'})",
            "",
            f"Open requirements: {result['open_requirements']}.",
        ]
    lines += [
        "",
        "## Shared-location differences",
        "",
        "Each entry is second minus first. The combined mean MCSE assumes independent chain streams; it is displayed separately from the physical mesh/prior shift.",
        "",
        "| First | Second | Quantity | Mean difference | Combined mean MCSE |",
        "|---|---|---|---:|---:|",
    ]
    for i, (first_path, first) in enumerate(records):
        if first["status"] != "sampled":
            continue
        for second_path, second in records[i + 1 :]:
            if second["status"] != "sampled" or first["truth"] != second["truth"]:
                continue
            mesh_pair = (
                first["prior_case"] == second["prior_case"] == "baseline"
                and first["nmesh"] != second["nmesh"]
            )
            prior_pair = (
                first["nmesh"] == second["nmesh"] == 16
                and (
                    first["prior_case"] == "baseline"
                    or second["prior_case"] == "baseline"
                )
                and first["prior_case"] != second["prior_case"]
            )
            if not (mesh_pair or prior_pair):
                continue
            for a, b in zip(
                first["shared_locations"][:5], second["shared_locations"][:5]
            ):
                lines.append(
                    f"| {first_path.parent.name} | {second_path.parent.name} | {a['label']} | {b['mean']-a['mean']:.8g} | {math.hypot(a['mean_mcse'], b['mean_mcse']):.4g} |"
                )
    lines += [
        "",
        "## Acceptance and efficiency comparisons",
        "",
        "Intervals use independent seeded 500-block resamples for each posterior. Efficiency ratios use the same physical summary in both calculations; they refer to measured retained-trajectory costs.",
        "",
        "| First | Second | Comparison | 95% block-resampling interval |",
        "|---|---|---|---|",
    ]
    for i, (first_path, first) in enumerate(records):
        if first["status"] != "sampled":
            continue
        for second_path, second in records[i + 1 :]:
            if second["status"] != "sampled" or first["truth"] != second["truth"]:
                continue
            mesh_pair = (
                first["prior_case"] == second["prior_case"] == "baseline"
                and first["nmesh"] != second["nmesh"]
            )
            prior_pair = (
                first["nmesh"] == second["nmesh"] == 16
                and (
                    first["prior_case"] == "baseline"
                    or second["prior_case"] == "baseline"
                )
                and first["prior_case"] != second["prior_case"]
            )
            if not (mesh_pair or prior_pair):
                continue
            uncertainty = _comparison_uncertainty(
                first["uncertainty"], second["uncertainty"]
            )
            if uncertainty["status"] != "resolved":
                lines.append(
                    f"| {first_path.parent.name} | {second_path.parent.name} | acceptance and efficiency | unresolved |"
                )
                continue
            lines.append(
                f"| {first_path.parent.name} | {second_path.parent.name} | acceptance second minus first | {uncertainty['acceptance_difference_interval']} |"
            )
            for index, item in enumerate(first["shared_locations"][:5]):
                for unit in ("second", "solve"):
                    interval = uncertainty[f"ess_per_{unit}_ratio_intervals"][index]
                    lines.append(
                        f"| {first_path.parent.name} | {second_path.parent.name} | {item['label']}: ESS/{unit} second / first | {interval} |"
                    )
    destination.write_text("\n".join(lines) + "\n")


def _comparison_uncertainty(first, second):
    if any(item["status"] != "resolved" for item in (first, second)):
        return {"status": "unresolved"}
    if first["seed"] == second["seed"]:
        return {
            "status": "unresolved",
            "reason": "Comparison requires independently seeded block resamples.",
        }
    result = {
        "status": "resolved",
        "acceptance_difference_interval": np.quantile(
            np.asarray(second["acceptance_resamples"])
            - np.asarray(first["acceptance_resamples"]),
            [0.025, 0.975],
        ).tolist(),
    }
    for unit in ("second", "solve"):
        ratio = (
            np.asarray(second[f"ess_per_{unit}_resamples"])[:, :5]
            / np.asarray(first[f"ess_per_{unit}_resamples"])[:, :5]
        )
        result[f"ess_per_{unit}_ratio_intervals"] = np.quantile(
            ratio, [0.025, 0.975], axis=0
        ).T.tolist()
    return result


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--render-figures":
        raise SystemExit(
            "This internal entry point only renders saved physical figures."
        )
    _render_study_figures(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()))
