"""Convergence evidence for physical conductivity and thermometer temperatures.

Statistics use ArviZ Stats' chain-preserving array interface. Geometry is
reconstructed lazily from the frozen experiment; no heat solve, likelihood,
random stream, or continuation state is changed. The JSON report is advisory.
"""

# type imports
from dataclasses import asdict, dataclass, replace
import hashlib
import json

# numerical imports
import numpy as np

# local imports
from posterior_store import _atomic_file, _problem_content_digest

PACKAGE_VERSION = "1.3.3"
POLICY_VERSION = 2
REPORT_FILE = "diagnostics.json"
STATISTICS = (
    "rank_split_rhat",
    "folded_split_rhat",
    "rhat",
    "bulk_ess",
    "tail_ess",
    "mean_mcse",
)


@dataclass(frozen=True)
class DiagnosticReport:
    r"""Saved convergence evidence, with chain counts and physical locations.

    ``per_quantity`` maps physical quantity names to statistics with a final
    node/sensor axis. Quantile MCSE has probability then location axes, in
    ``probabilities`` order. Rank R-hat is the maximum of the separately reported
    rank-normalized split and folded-split values. Tail ESS uses 5% and 95% tails.
    ``common_draws`` labels the inspected common prefix; unequal or unfinished
    chains never qualify the requested full calculation. Analytically fixed
    boundary sensors have undefined R-hat/ESS and exactly zero Monte Carlo errors.
    ``persisted`` says whether the advisory report was successfully saved.
    """

    status: str
    qualified: bool
    per_quantity: dict
    worst_locations: dict
    divergences: int
    counts: list
    package_version: str | None
    reasons: list
    probabilities: list
    common_draws: int
    requested_total: int
    identity: dict
    settings: dict
    persisted: bool = False


def qualify_diagnostics(
    *, chains, complete, rhat, bulk_ess, tail_ess, divergences, analytically_fixed
):
    r"""Apply the accepted strict R-hat/ESS and complete-chain criteria.

    Parameters
    ----------
    chains : int
        Number of independent chains.
    complete : bool
        Whether every chain reached the full requested retained count.
    rhat, bulk_ess, tail_ess : array-like
        Combined-chain statistics with matching shapes.
    divergences : int
        Number of post-warmup divergences across all retained states.
    analytically_fixed : array-like of bool
        Exempt only quantities fixed by the prescribed-temperature operator.

    Returns
    -------
    tuple[bool, list[str]]
        Qualification and explicit reasons for failure. MCSE finiteness and
        accidental constant quantities are additionally checked by the report.
    """
    arrays = [np.asarray(x, dtype=float) for x in (rhat, bulk_ess, tail_ess)]
    fixed = np.asarray(analytically_fixed, dtype=bool)
    reasons = []
    if chains < 4:
        reasons.append("At least four chains are required.")
    if not complete:
        reasons.append("The full requested calculation is incomplete.")
    if divergences != 0:
        reasons.append("Post-warmup divergences must be zero.")
    if any(a.shape != fixed.shape for a in arrays):
        reasons.append("Diagnostic quantities and fixed-sensor flags do not match.")
    elif fixed.size == 0:
        reasons.append("No diagnostic quantities were supplied.")
    else:
        active = ~fixed
        for name, values, passes in (
            ("R-hat", arrays[0], arrays[0] < 1.01),
            ("bulk ESS", arrays[1], arrays[1] >= 400),
            ("tail ESS", arrays[2], arrays[2] >= 400),
        ):
            if not np.isfinite(values[active]).all():
                reasons.append(f"Missing or nonfinite {name}.")
            elif not passes[active].all():
                reasons.append(f"{name} fails the accepted convergence threshold.")
    return not reasons, reasons


def _load_backend():
    from arviz_stats import __version__
    from arviz_stats.base import array_stats

    return array_stats, __version__


def _settings(probabilities):
    return dict(
        policy_version=POLICY_VERSION,
        package_version=PACKAGE_VERSION,
        chain_axis=0,
        draw_axis=1,
        tail_probabilities=[0.05, 0.95],
        quantile_probabilities=list(probabilities),
        rhat_limit=1.01,
        minimum_ess=400,
        minimum_chains=4,
    )


def _run_content_digest(samples, metadata):
    """Bind the report to this calculation's physical and random chain states.

    Typed array bytes and canonical mapping order are independent of path,
    archive timestamps, and mapping insertion order. Reading only initial and
    current checkpoints costs O(chains times field size), preserving cheap
    qualification lookup and legitimate backups/relocations.
    """

    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            return [
                "array" if isinstance(value, np.ndarray) else "numpy_scalar",
                array.dtype.str,
                list(array.shape),
                hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            ]
        if isinstance(value, dict):
            return ["mapping", [[key, encode(value[key])] for key in sorted(value)]]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(item) for item in value]]
        if isinstance(value, float):
            # Preserve negative zero and explicit nonfinite metadata without
            # relying on nonstandard JSON NaN/Infinity literals.
            return ["float", value.hex()]
        if value is None or type(value) in (str, int, bool):
            return [type(value).__name__, value]
        raise TypeError(f"Unsupported saved provenance type: {type(value).__name__}.")

    checkpoints = []
    for chain in range(metadata["chains"]):
        evidence = {}
        for name, read in (
            ("initial", samples._store.initial_checkpoint),
            ("current", samples._store.checkpoint),
        ):
            try:
                evidence[name] = {"present": True, "state": asdict(read(chain))}
            except KeyError:
                evidence[name] = {"present": False}
        checkpoints.append(evidence)
    canonical = encode(dict(metadata=metadata, checkpoints=checkpoints))
    payload = json.dumps(
        canonical, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _snapshot(samples):
    metadata = samples.metadata
    return dict(
        counts=samples._counts().tolist(),
        requested_total=int(metadata["total_draws"]),
        chains=int(metadata["chains"]),
        problem_id=samples._record.problem_id,
        numerical_id=samples._record.numerical_id,
        problem_digest=_problem_content_digest(samples._record),
        run_digest=_run_content_digest(samples, metadata),
    )


def _clip(polygon, axis, dividing_line):
    """Intersect a convex cell polygon with the lower physical half-plane."""
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        inside_a, inside_b = a[axis] <= dividing_line, b[axis] <= dividing_line
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            fraction = (dividing_line - a[axis]) / (b[axis] - a[axis])
            result.append(a + fraction * (b - a))
    return np.asarray(result).reshape(-1, 2)


def _prescribed_geometry(coordinates, cells, boundary_dofs, points):
    """Prove fixed P1/Q1 traces using exact stored-coordinate geometry.

    Tiny interpolation coefficients alone cannot distinguish a boundary sensor
    from a near-interior sensor. Exact rational orientation of the stored floats
    tests constrained nodes, constrained mesh edges, and fully constrained cells.
    Bounding boxes reject most candidates before rational arithmetic is needed.
    """
    from fractions import Fraction

    prescribed = np.zeros(len(coordinates), dtype=bool)
    prescribed[boundary_dofs] = True
    edges, interiors = set(), []
    for cell in cells:
        vertices = coordinates[cell]
        centered = vertices - vertices.mean(axis=0)
        order = np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))
        polygon = cell[order]
        for a, b in zip(polygon, np.roll(polygon, -1)):
            if prescribed[a] and prescribed[b]:
                edges.add(tuple(sorted((int(a), int(b)))))
        if prescribed[cell].all():
            interiors.append(coordinates[polygon])

    def rational(point):
        return tuple(Fraction(float(x)) for x in point)

    def orientation(a, b, point):
        a, b, point = rational(a), rational(b), rational(point)
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])

    def in_box(point, vertices):
        return bool(
            np.all(point >= vertices.min(axis=0))
            and np.all(point <= vertices.max(axis=0))
        )

    result = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if np.any(np.all(coordinates[prescribed] == point, axis=1)):
            result[i] = True
            continue
        for a, b in edges:
            vertices = coordinates[[a, b]]
            if in_box(point, vertices) and orientation(*vertices, point) == 0:
                result[i] = True
                break
        if result[i]:
            continue
        for polygon in interiors:
            if not in_box(point, polygon):
                continue
            signs = [
                orientation(a, b, point)
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
            ]
            if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
                result[i] = True
                break
    return result


def _physical_observables(record):
    """Integrate the authoritative P1/Q1 interpolation over physical regions.

    Three degree-two quadrature points per subtriangle integrate P1 and affine
    Q1 exactly. Polygon clipping treats cells crossed by a physical dividing
    line. All basis evaluations use SensorOperator; it also identifies sensors
    on analytically constrained traces, with roundoff-only free-DOF support.
    Returns normalized mean/contrast weights in canonical node order, then the
    prescribed-temperature sensor flags. No heat equation is solved.
    """
    from dolfinx import fem
    from heat_model import FrozenHeatModel, canonical_dof_indices
    from temperature_observations import SensorOperator

    arrays = record.arrays
    xy = arrays["coordinates"]
    model = FrozenHeatModel(
        xy,
        arrays["cells"],
        record.metadata["model"]["cell_type"],
        arrays["source"],
        arrays["boundary_dofs"],
        arrays["boundary_values"],
    )
    V = fem.functionspace(model._build_mesh(), ("Lagrange", 1))
    order = canonical_dof_indices(V)
    mid = (xy.min(axis=0) + xy.max(axis=0)) / 2
    integrals = np.zeros((3, len(xy)))
    points, areas, groups = [], [], []
    barycentric = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )

    def accumulate():
        if not points:
            return
        B = SensorOperator(V, np.asarray(points)).matrix
        area, group = np.asarray(areas), np.asarray(groups)
        for i in range(3):
            integrals[i] += np.asarray(B.T @ (area * (group == i))).reshape(-1)
        points.clear()
        areas.clear()
        groups.clear()

    for cell in arrays["cells"]:
        vertices = xy[cell]
        relative = vertices - vertices.mean(axis=0)
        vertices = vertices[np.argsort(np.arctan2(relative[:, 1], relative[:, 0]))]
        regions = (vertices, _clip(vertices, 0, mid[0]), _clip(vertices, 1, mid[1]))
        for region, polygon in enumerate(regions):
            for j in range(1, len(polygon) - 1):
                triangle = polygon[[0, j, j + 1]]
                u, v = triangle[1:] - triangle[0]
                area = abs(u[0] * v[1] - u[1] * v[0]) / 2
                if area == 0:
                    continue
                points.extend(barycentric @ triangle)
                areas.extend([area / 3] * 3)
                groups.extend([region] * 3)
                if len(points) >= 4096:
                    accumulate()
    accumulate()
    full, left, bottom = integrals[:, order]
    right, top = full - left, full - bottom
    regions = (full, left, right, bottom, top)
    if any(not np.isfinite(w).all() or w.sum() <= 0 for w in regions):
        raise ValueError("Physical half-domain areas must be finite and positive.")
    weights = np.array(
        [
            full / full.sum(),
            left / left.sum() - right / right.sum(),
            bottom / bottom.sum() - top / top.sum(),
        ]
    )
    sensor = SensorOperator(V, arrays["sensor_points"]).matrix
    free = np.ones(len(xy), dtype=bool)
    free[order[arrays["boundary_dofs"]]] = False
    analytic = _prescribed_geometry(
        xy, arrays["cells"], arrays["boundary_dofs"], arrays["sensor_points"]
    )
    residual = np.asarray(abs(sensor[:, free]).sum(axis=1)).reshape(-1)
    scale = np.asarray(abs(sensor).sum(axis=1)).reshape(-1)
    fixed = analytic & (scale > 0) & (residual <= 64 * np.finfo(float).eps * scale)
    return weights, fixed


def _quantity_statistics(backend, values, probabilities, fixed):
    """Evaluate a bounded block, retaining chain and draw axes 0 and 1."""
    count = values.shape[-1]
    statistics = {name: np.full(count, np.nan) for name in STATISTICS}
    statistics["quantile_mcse"] = np.full((len(probabilities), count), np.nan)
    statistics["analytically_fixed"] = np.asarray(fixed, dtype=bool)
    constant = np.zeros(count, dtype=bool)
    if values.shape[1]:
        constant = np.all(values == values[0, 0], axis=(0, 1))
    statistics["constant"] = constant
    # Fixed sensors require no stochastic error estimate. A changing recorded
    # temperature cannot be excused by this analytic designation.
    consistent_fixed = fixed & constant
    statistics["mean_mcse"][consistent_fixed] = 0.0
    statistics["quantile_mcse"][:, consistent_fixed] = 0.0
    if values.shape[1] < 4:
        return statistics
    active = ~fixed & ~constant & np.isfinite(values).all(axis=(0, 1))
    if not active.any():
        return statistics
    block = values[..., active]
    kw = dict(chain_axis=0, draw_axis=1)
    for name, method in (
        ("rank_split_rhat", "z_scale"),
        ("folded_split_rhat", "folded"),
        ("rhat", "rank"),
    ):
        statistics[name][active] = backend.rhat(block, method=method, **kw)
    statistics["bulk_ess"][active] = backend.ess(block, method="bulk", **kw)
    statistics["tail_ess"][active] = backend.ess(
        block, method="tail", prob=(0.05, 0.95), **kw
    )
    statistics["mean_mcse"][active] = backend.mcse(block, method="mean", **kw)
    for i, probability in enumerate(probabilities):
        # Sample extrema are not regular posterior quantiles with the accepted
        # quantile MCSE interpretation. Keep raw extrema available in summary.
        if 0 < probability < 1:
            statistics["quantile_mcse"][i, active] = backend.mcse(
                block, method="quantile", prob=probability, **kw
            )
    return statistics


def _lists(value):
    """JSON-compatible evidence; undefined diagnostics are explicit nulls."""
    if isinstance(value, np.ndarray):
        return _lists(value.tolist())
    if isinstance(value, dict):
        return {k: _lists(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_lists(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _worst(quantities, probabilities):
    result = {}
    undefined = {}
    for statistic, maximum in (
        ("rhat", True),
        ("bulk_ess", False),
        ("tail_ess", False),
        ("mean_mcse", True),
    ):
        candidates = []
        undefined[statistic] = []
        for name, record in quantities.items():
            values = np.asarray(record[statistic], dtype=float)
            fixed = np.asarray(record["analytically_fixed"], dtype=bool)
            for index in np.flatnonzero(~fixed & ~np.isfinite(values)):
                undefined[statistic].append(
                    dict(
                        quantity=name,
                        index=int(index),
                        location=record["locations"][index],
                    )
                )
            for index in np.flatnonzero(~fixed & np.isfinite(values)):
                candidates.append(
                    dict(
                        quantity=name,
                        index=int(index),
                        location=record["locations"][index],
                        value=float(values[index]),
                    )
                )
        result[statistic] = (
            (max if maximum else min)(candidates, key=lambda c: c["value"])
            if candidates
            else None
        )
    result["undefined"] = undefined
    result["quantile_mcse"] = []
    for i, probability in enumerate(probabilities):
        candidates = []
        missing = []
        for name, record in quantities.items():
            values = np.asarray(record["quantile_mcse"], dtype=float)[i]
            for index in np.flatnonzero(~np.asarray(record["analytically_fixed"])):
                evidence = dict(
                    quantity=name, index=int(index), location=record["locations"][index]
                )
                if np.isfinite(values[index]):
                    candidates.append(evidence | {"value": float(values[index])})
                else:
                    missing.append(evidence)
        result["quantile_mcse"].append(
            dict(
                probability=probability,
                worst=max(candidates, key=lambda c: c["value"]) if candidates else None,
                undefined=missing,
            )
        )
    return result


def _persist(samples, report):
    if _snapshot(samples) != report.identity:
        return replace(
            report,
            status="incomplete",
            qualified=False,
            reasons=report.reasons
            + [
                "Saved counts or requested target changed during diagnostics; recompute."
            ],
        )
    saved = replace(report, persisted=True)
    try:
        _atomic_file(
            samples.output / REPORT_FILE,
            json.dumps(_lists(asdict(saved)), allow_nan=False).encode(),
        )
    except OSError as error:
        return replace(
            report,
            reasons=report.reasons
            + [f"Diagnostics were computed but could not be saved: {error}"],
        )
    if _snapshot(samples) != report.identity:
        return replace(
            saved,
            status="incomplete",
            qualified=False,
            reasons=report.reasons
            + [
                "Saved counts or requested target changed while saving diagnostics; recompute."
            ],
        )
    return saved


def diagnose(samples, *, node_chunk_size=64, quantiles=(0.05, 0.5, 0.95)):
    """Compute and save physical diagnostics without altering any chain state."""
    from posterior_samples import _index, _probabilities

    width = _index(node_chunk_size, "node_chunk_size", minimum=1)
    probabilities = _probabilities(quantiles).tolist()
    snapshot = _snapshot(samples)
    counts = snapshot["counts"]
    common = min(counts)
    complete = all(c == snapshot["requested_total"] for c in counts)
    reasons = []
    if not complete:
        reasons.append(
            f"Inspection uses common prefix of {common} draws per chain; the requested calculation is incomplete."
        )
    divergences, missing_divergences = 0, False
    for chain, count in enumerate(counts):
        for record in samples._store.records(chain, stop=count):
            flag = record.statistics.get("divergent")
            if not isinstance(flag, (bool, np.bool_)):
                missing_divergences = True
            else:
                divergences += int(flag)
    if missing_divergences:
        reasons.append("Missing post-warmup divergence evidence in retained records.")
    report = DiagnosticReport(
        "incomplete" if not complete else "unqualified",
        False,
        {},
        {},
        divergences,
        counts,
        None,
        reasons,
        probabilities,
        common,
        snapshot["requested_total"],
        snapshot,
        _settings(probabilities),
    )
    try:
        backend, version = _load_backend()
    except ImportError:
        return _persist(
            samples,
            replace(
                report,
                status="unavailable",
                reasons=reasons
                + [
                    "Install the diagnostics extra: pip install 'steady-2d-heat-inverse[diagnostics]' (Python 3.12+). "
                ],
            ),
        )
    report = replace(report, package_version=version)
    if version != PACKAGE_VERSION:
        return _persist(
            samples,
            replace(
                report,
                status="unavailable",
                reasons=reasons
                + [
                    f"Diagnostics require qualified arviz-stats=={PACKAGE_VERSION}; found {version}."
                ],
            ),
        )
    weights, fixed = _physical_observables(samples._record)
    quantities = {}

    def evaluate(name, size, locations, flags, read):
        parts = []
        for begin in range(0, size, width):
            end = min(begin + width, size)
            values = np.empty((len(counts), common, end - begin))
            for chain in range(len(counts)):
                for draw, record in enumerate(
                    samples._store.records(chain, stop=common)
                ):
                    values[chain, draw] = read(record, begin, end)
            parts.append(
                _quantity_statistics(backend, values, probabilities, flags[begin:end])
            )
        merged = {
            key: np.concatenate([part[key] for part in parts], axis=-1)
            for key in parts[0]
        }
        merged["locations"] = locations
        quantities[name] = merged

    nodes = len(samples.coordinates)
    evaluate(
        "conductivity",
        nodes,
        samples.coordinates.tolist(),
        np.zeros(nodes, dtype=bool),
        lambda r, a, b: r.h[a:b],
    )
    for index, name in enumerate(("area_mean", "left_minus_right", "bottom_minus_top")):
        evaluate(
            name,
            1,
            [None],
            np.array([False]),
            lambda r, a, b: np.array([r.h @ weights[index]]),
        )
    evaluate(
        "sensor_temperature",
        len(fixed),
        samples._record.arrays["sensor_points"].tolist(),
        fixed,
        lambda r, a, b: r.sensor_temperature[a:b],
    )
    if any(p in (0.0, 1.0) for p in probabilities):
        reasons.append(
            "MCSE of quantile endpoints at probability 0 or 1 is undefined for this qualification."
        )
    for name, values in quantities.items():
        _, failures = qualify_diagnostics(
            chains=len(counts),
            complete=complete,
            divergences=divergences,
            **{
                k: values[k]
                for k in ("rhat", "bulk_ess", "tail_ess", "analytically_fixed")
            },
        )
        reasons.extend(f"{name}: {failure}" for failure in failures)
        active = ~values["analytically_fixed"]
        if np.any(values["constant"] & active):
            reasons.append(f"{name}: inferred quantity is accidentally constant.")
        if np.any(values["analytically_fixed"] & ~values["constant"]):
            reasons.append(
                f"{name}: recorded sensor contradicts its prescribed fixed temperature."
            )
        for statistic in (
            "rank_split_rhat",
            "folded_split_rhat",
            "mean_mcse",
            "quantile_mcse",
        ):
            if not np.isfinite(values[statistic][..., active]).all():
                reasons.append(f"{name}: nonfinite {statistic} (MCSE or split R-hat).")
    qualified = not reasons
    report = replace(
        report,
        status=(
            "qualified" if qualified else ("unqualified" if complete else "incomplete")
        ),
        qualified=qualified,
        per_quantity=_lists(quantities),
        worst_locations=_worst(quantities, probabilities),
        divergences=divergences,
        reasons=reasons,
    )
    return _persist(samples, report)


def matching_qualification(samples, probabilities, counts):
    """Read only a complete report matching this physical history and intervals."""
    try:
        before = _snapshot(samples)
        document = json.loads((samples.output / REPORT_FILE).read_text())
        recorded = document["probabilities"]
        valid = (
            document["qualified"] is True
            and document["status"] == "qualified"
            and document["persisted"] is True
            and document["package_version"] == PACKAGE_VERSION
            and document["settings"] == _settings(recorded)
            and document["identity"] == before
            and document["counts"] == list(counts)
            and document["requested_total"] == before["requested_total"]
            and all(c == before["requested_total"] for c in before["counts"])
            and all(p in recorded for p in probabilities)
        )
        return bool(valid and _snapshot(samples) == before)
    except (OSError, ValueError, TypeError, KeyError):
        return False
