"""Qualification of saved physical fields; manufactured draws test statistics only."""

from dataclasses import replace
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from posterior_diagnostics import (
    qualify_diagnostics,
    _physical_observables,
    _quantity_statistics,
    _prescribed_geometry,
)
from posterior_samples import PosteriorSamples
from posterior_store import ChainCheckpoint, RunStore, TrajectoryRecord
from tests._hmc_helpers import make_forward, _release_problem


def require_backend():
    try:
        from arviz_stats.base import array_stats

        return array_stats
    except ImportError:
        if os.environ.get("S2DHI_HMC_QUALIFICATION") == "1":
            raise
        pytest.skip("optional diagnostics extra is absent")


def test_qualification_thresholds_are_strict_and_missing_values_fail():
    base = dict(
        chains=4,
        complete=True,
        rhat=np.array([1.009]),
        bulk_ess=np.array([400.0]),
        tail_ess=np.array([400.0]),
        divergences=0,
        analytically_fixed=np.array([False]),
    )
    assert qualify_diagnostics(**base)[0]
    for changes in (
        {"rhat": np.array([1.01])},
        {"tail_ess": np.array([399.9])},
        {"bulk_ess": np.array([399.9])},
        {"rhat": np.array([np.nan])},
        {"bulk_ess": np.array([np.inf])},
        {"tail_ess": np.array([np.nan])},
        {"divergences": 1},
        {"chains": 1},
        {"complete": False},
    ):
        assert not qualify_diagnostics(**(base | changes))[0]
    fixed = base | {
        "rhat": np.array([np.nan]),
        "bulk_ess": np.array([np.nan]),
        "tail_ess": np.array([np.nan]),
        "analytically_fixed": np.array([True]),
    }
    assert qualify_diagnostics(**fixed)[0]


def make_record(nmesh=3, cell="triangle", points=None, bottom=300.0):
    from conductivity_prior import GaussianLogConductivity
    from heat_inference import HeatInverseProblem
    from temperature_observations import TemperatureObservations

    if points is None:
        points = np.array([[0.25, 0.0], [0.37, 0.61]])
    forward = make_forward(nmesh=nmesh, cell=cell, bottom=bottom)
    try:
        with HeatInverseProblem(
            forward=forward,
            observations=TemperatureObservations(
                points, np.full(len(points), bottom), 0.05
            ),
            prior=GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1),
        ) as problem:
            return problem.to_record()
    finally:
        _release_problem(forward.problem)


def save_manufactured(
    path, *, lengths=(1024,) * 4, divergent=False, seed=513, fixed_offset=0.0
):
    """Random physical arrays exercise bookkeeping, not posterior correctness."""
    record = make_record(nmesh=1)
    rng = np.random.default_rng(seed)
    h = np.exp(0.1 * rng.normal(size=(len(lengths), max(lengths), 4)))
    sensor = 300 + rng.normal(size=(len(lengths), max(lengths), 2))
    sensor[:, :, 0] = 300 + fixed_offset
    initial = [
        ChainCheckpoint(
            c,
            "sample",
            0,
            0,
            {"z": np.zeros(4)},
            {"h": h[c, 0], "sensor_temperature": sensor[c, 0]},
            {"position": np.zeros(4)},
            {"key": np.array([c, seed])},
            {},
            0.1,
            {},
        )
        for c in range(len(lengths))
    ]
    with RunStore.create(
        path,
        problem=record,
        run_metadata={
            "chains": len(lengths),
            "total_draws": max(lengths),
            "warmup": 0,
            "master_seed": seed,
        },
    ) as store:
        store.initialize_chains(initial)
        for c, count in enumerate(lengths):
            cp = initial[c]
            for d in range(count):
                cp = replace(
                    cp,
                    draws_done=d + 1,
                    physical_state={"h": h[c, d], "sensor_temperature": sensor[c, d]},
                )
                store.commit_trajectory(
                    TrajectoryRecord(
                        c,
                        "sample",
                        d,
                        cp.inferred_coordinates,
                        h[c, d],
                        sensor[c, d],
                        {"divergent": bool(divergent and c == 0 and d == count - 1)},
                    ),
                    cp,
                )
    return h, sensor


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    path = tmp_path_factory.mktemp("diagnostics") / "run"
    h, sensor = save_manufactured(path)
    return path, h, sensor


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_exact_physical_cut_cell_weights_and_prescribed_sensor(cell):
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_vector
    from petsc4py import PETSc
    import ufl
    from heat_model import FrozenHeatModel, canonical_dof_indices

    record = make_record(nmesh=3, cell=cell)
    weights, fixed = _physical_observables(record)
    xy = record.arrays["coordinates"]
    model = FrozenHeatModel(
        xy,
        record.arrays["cells"],
        cell,
        record.arrays["source"],
        record.arrays["boundary_dofs"],
        record.arrays["boundary_values"],
    )
    V = fem.functionspace(model._build_mesh(), ("Lagrange", 1))
    mass = assemble_vector(fem.form(ufl.TestFunction(V) * ufl.dx))
    try:
        mass.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        expected = mass.array[canonical_dof_indices(V)].copy()
    finally:
        mass.destroy()
    assert np.allclose(weights[0], expected / expected.sum(), rtol=0, atol=1e-14)
    linear = 1 + 2 * xy[:, 0] + 3 * xy[:, 1]
    assert np.allclose(weights @ linear, [3.5, -1.0, -1.5], rtol=0, atol=1e-13)
    if cell == "quadrilateral":
        bilinear = 1 + xy[:, 0] * xy[:, 1]
        assert np.allclose(weights @ bilinear, [1.25, -0.25, -0.25], rtol=0, atol=1e-13)
    assert np.array_equal(fixed, [True, False])


def direct_statistics(values, probabilities):
    array_stats = require_backend()
    kw = dict(chain_axis=0, draw_axis=1)
    return {
        "rank_split_rhat": array_stats.rhat(values, method="z_scale", **kw),
        "folded_split_rhat": array_stats.rhat(values, method="folded", **kw),
        "rhat": array_stats.rhat(values, method="rank", **kw),
        "bulk_ess": array_stats.ess(values, method="bulk", **kw),
        "tail_ess": array_stats.ess(values, method="tail", prob=(0.05, 0.95), **kw),
        "mean_mcse": array_stats.mcse(values, method="mean", **kw),
        "quantile_mcse": np.array(
            [
                array_stats.mcse(values, method="quantile", prob=p, **kw)
                for p in probabilities
            ]
        ),
    }


def test_every_physical_quantity_matches_pinned_package(saved, monkeypatch):
    require_backend()
    path, h, sensor = saved
    with PosteriorSamples.open(path) as samples:
        monkeypatch.setattr(
            samples, "load_draws", lambda **kw: pytest.fail("whole arrays loaded")
        )
        report = samples.diagnostics(node_chunk_size=2, quantiles=(0.1, 0.9))
        assert report.package_version == "1.3.3"
        assert report.status == "qualified", report.reasons
        assert report.qualified
        fixed_evidence = report.fixed_sensor_consistency[0]
        assert fixed_evidence["sensor"] == 0
        assert fixed_evidence["boundary_reference"] == 300.0
        assert fixed_evidence["maximum_absolute_residual"] == 0.0
        assert fixed_evidence["status"] == "exact_agreement"
        assert report.probabilities == [0.1, 0.9]
        assert report.counts == [1024] * 4
        weights, fixed = _physical_observables(samples._record)
        fields = {"conductivity": h, "sensor_temperature": sensor}
        for i, name in enumerate(("area_mean", "left_minus_right", "bottom_minus_top")):
            fields[name] = (h @ weights[i])[..., None]
        assert set(report.per_quantity) == set(fields)
        for name, values in fields.items():
            got = report.per_quantity[name]
            # Prescribed boundary temperatures have exact zero Monte Carlo errors.
            mask = (
                ~fixed
                if name == "sensor_temperature"
                else np.ones(values.shape[-1], dtype=bool)
            )
            expected = direct_statistics(values[..., mask], (0.1, 0.9))
            for statistic, answer in expected.items():
                assert np.allclose(
                    np.asarray(got[statistic], dtype=float)[..., mask],
                    answer,
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
        assert report.per_quantity["sensor_temperature"]["analytically_fixed"] == [
            True,
            False,
        ]
        assert np.array_equal(
            report.per_quantity["conductivity"]["locations"], samples.coordinates
        )
        assert np.array_equal(
            report.per_quantity["sensor_temperature"]["locations"],
            samples._record.arrays["sensor_points"],
        )
        assert report.worst_locations["rhat"]["quantity"] in fields
        assert samples.summary(quantiles=(0.1, 0.9)).qualified
        assert not samples.summary().qualified
    with PosteriorSamples.open(path) as reopened:
        assert reopened.summary(quantiles=(0.1, 0.9)).qualified


@pytest.mark.parametrize("kind", ["separated", "correlated", "constant"])
def test_nonconverged_draws_cannot_pass(kind):
    rng = np.random.default_rng(95)
    values = rng.normal(size=(4, 1024, 1))
    if kind == "separated":
        values += np.arange(4)[:, None, None] * 4
    elif kind == "correlated":
        for d in range(1, 1024):
            values[:, d] = 0.995 * values[:, d - 1] + 0.1 * values[:, d]
    else:
        values[:] = 2
    array_stats = require_backend()
    stats = _quantity_statistics(array_stats, values, [0.05, 0.95], np.array([False]))
    assert not qualify_diagnostics(
        chains=4,
        complete=True,
        divergences=0,
        analytically_fixed=np.array([False]),
        **{k: stats[k] for k in ("rhat", "bulk_ess", "tail_ess")},
    )[0]


def test_partial_prefix_and_divergence_over_full_retained_history(tmp_path):
    require_backend()
    path = tmp_path / "partial"
    save_manufactured(path, lengths=(12, 8, 9, 10), divergent=True)
    with PosteriorSamples.open(path) as samples:
        report = samples.diagnostics()
        assert report.status == "incomplete"
        assert report.common_draws == 8
        assert report.counts == [12, 8, 9, 10]
        assert report.divergences == 1
        assert not report.qualified
        assert any("common prefix" in r for r in report.reasons)
        assert not samples.summary().qualified


def test_extension_invalidates_report_before_next_draw_and_recomputes(saved, tmp_path):
    require_backend()
    path = tmp_path / "extended"
    with RunStore.open(saved[0]) as store:
        store.backup(path)
    with PosteriorSamples.open(path) as samples:
        first = samples.diagnostics()
        assert first.qualified
        with RunStore.open(path, writable=True) as writer:
            writer.update_target(1025)
        assert not samples.summary().qualified
        second = samples.diagnostics()
        assert second.status == "incomplete"
        assert second.requested_total == 1025
        assert second.counts == first.counts


def test_missing_extra_nonfinite_statistics_and_endpoint_mcse(saved, monkeypatch):
    import posterior_diagnostics as diagnostics

    path, _, _ = saved
    with PosteriorSamples.open(path) as samples:
        with monkeypatch.context() as patch:
            patch.setattr(
                diagnostics,
                "_load_backend",
                lambda: (_ for _ in ()).throw(ImportError("absent")),
            )
            report = samples.diagnostics()
            assert report.status == "unavailable"
            assert any("[diagnostics]" in r for r in report.reasons)
        require_backend()
        report = samples.diagnostics(quantiles=(0.0, 1.0))
        assert not report.qualified
        assert any("endpoint" in r.lower() for r in report.reasons)
        assert not samples.summary(quantiles=(0.0, 1.0)).qualified
        from arviz_stats.base import array_stats

        with monkeypatch.context() as patch:
            patch.setattr(
                array_stats, "mcse", lambda ary, **kw: np.full(ary.shape[-1], np.nan)
            )
            report = samples.diagnostics()
            assert not report.qualified
            assert any("MCSE" in r for r in report.reasons)


def test_diagnostics_needs_no_jax_or_plotting(saved):
    require_backend()
    path, _, _ = saved
    script = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax','blackjax','matplotlib','pyvista'}:
            raise ImportError('blocked '+fullname)
sys.meta_path.insert(0, Block())
from posterior_samples import PosteriorSamples
with PosteriorSamples.open(sys.argv[1]) as samples:
    report = samples.diagnostics(node_chunk_size=1)
    assert report.status in {'qualified', 'incomplete'}, report.reasons
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_persistence_failure_keeps_computed_evidence(saved, monkeypatch):
    require_backend()
    import posterior_diagnostics as diagnostics

    def fail(*args):
        raise PermissionError("read-only report destination")

    monkeypatch.setattr(diagnostics, "_atomic_file", fail)
    with PosteriorSamples.open(saved[0]) as samples:
        report = samples.diagnostics()
        assert report.qualified
        assert not report.persisted
        assert report.per_quantity["conductivity"]
        assert any("could not be saved" in r for r in report.reasons)


def test_extension_during_diagnostics_cannot_qualify(saved, tmp_path, monkeypatch):
    require_backend()
    import posterior_diagnostics as diagnostics

    path = tmp_path / "concurrent"
    with RunStore.open(saved[0]) as store:
        store.backup(path)
    original = diagnostics._quantity_statistics
    extended = False

    def extending(*args, **kwargs):
        nonlocal extended
        if not extended:
            with RunStore.open(path, writable=True) as writer:
                writer.update_target(1025)
            extended = True
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "_quantity_statistics", extending)
    with PosteriorSamples.open(path) as samples:
        report = samples.diagnostics()
        assert report.status == "incomplete"
        assert not report.qualified
        assert not report.persisted
        assert not samples.summary().qualified


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_fixed_sensor_geometry_excludes_near_interior_points(cell):
    points = np.array(
        [[0.25, 0.0], [0.25, 1e-20], [0.25, 1e-15], [0.25, 1e-12], [0.37, 0.61]]
    )
    record = make_record(nmesh=3, cell=cell, points=points)
    _, fixed = _physical_observables(record)
    assert np.array_equal(fixed, [True, False, False, False, False])
    stats = _quantity_statistics(None, np.ones((4, 4, 1)), [0.05, 0.95], fixed[1:2])
    assert not qualify_diagnostics(
        chains=4,
        complete=True,
        divergences=0,
        analytically_fixed=fixed[1:2],
        **{name: stats[name] for name in ("rhat", "bulk_ess", "tail_ess")},
    )[0]


def test_exact_prescribed_node_edge_and_cell_membership():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    points = np.array([[0.0, 0.0], [0.5, 0.0], [0.2, 0.2], [0.8, 0.8], [0.5, 1e-20]])
    assert np.array_equal(
        _prescribed_geometry(xy, cells, np.array([0]), points),
        [True, False, False, False, False],
    )
    assert np.array_equal(
        _prescribed_geometry(xy, cells, np.array([0, 1]), points),
        [True, True, False, False, False],
    )
    assert np.array_equal(
        _prescribed_geometry(xy, cells, np.array([0, 1, 2]), points),
        [True, True, True, False, True],
    )
    # A slanted constrained edge uses exact orientation, with no spatial tolerance.
    points = np.array([[0.5, 0.5], [0.5, np.nextafter(0.5, 1.0)]])
    assert np.array_equal(
        _prescribed_geometry(xy, cells, np.array([1, 2]), points), [True, False]
    )


def test_report_binds_random_history_and_survives_backup(saved, tmp_path):
    require_backend()
    source = saved[0]
    with PosteriorSamples.open(source) as samples:
        assert samples.diagnostics().qualified
    foreign = tmp_path / "different_seed"
    save_manufactured(foreign, seed=817)
    shutil.copy2(source / "diagnostics.json", foreign / "diagnostics.json")
    with PosteriorSamples.open(foreign) as samples:
        assert (
            not samples.summary().qualified
        ), "a report from different random chains must not qualify this history"
    relocated = tmp_path / "relocated"
    with RunStore.open(source) as store:
        store.backup(relocated)
    shutil.copy2(source / "diagnostics.json", relocated / "diagnostics.json")
    with PosteriorSamples.open(relocated) as samples:
        assert samples.summary().qualified


def test_interrupted_initialization_has_explicit_incomplete_diagnostics(tmp_path):
    require_backend()
    path = tmp_path / "uninitialized"
    with RunStore.create(
        path,
        problem=make_record(nmesh=1),
        run_metadata={"chains": 4, "total_draws": 1024, "warmup": 0, "master_seed": 3},
    ):
        pass
    with PosteriorSamples.open(path) as samples:
        report = samples.diagnostics()
        assert report.status == "incomplete"
        assert not report.qualified
        assert report.counts == [0] * 4
        assert report.identity["run_digest"]


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
@pytest.mark.parametrize("bottom", [0.0, 300.0])
def test_real_fixed_boundary_predictions_have_zero_sampling_error(cell, bottom):
    from heat_inference import HeatInverseProblem

    record = make_record(nmesh=3, cell=cell, bottom=bottom)
    _, fixed = _physical_observables(record)
    readings = []
    with HeatInverseProblem.from_record(record) as problem:
        for conductivity in (1.0, 2.0, 0.1, 1e-6):
            _, sensors = problem.posterior.predict_h(
                np.full(problem.posterior.prior.dimension, conductivity)
            )
            readings.append(sensors[0])
    assert fixed[0]
    # These real predictions reproduce the measured interpolation roundoff.
    assert np.ptp(readings) > 0
    values = np.tile(np.asarray(readings)[None, :, None], (4, 1, 1))
    stats = _quantity_statistics(None, values, [0.05, 0.95], fixed[:1])
    assert stats["mean_mcse"][0] == 0
    assert np.array_equal(stats["quantile_mcse"][:, 0], [0.0, 0.0])
    assert not stats["constant"][0]  # Raw numerical evidence must remain visible.
    assert stats["raw_minimum"][0] == min(readings)
    assert stats["raw_maximum"][0] == max(readings)
    from posterior_diagnostics import _fixed_sensor_evidence

    _, _, reference = _physical_observables(record, include_fixed_temperatures=True)
    evidence = _fixed_sensor_evidence(record, stats, reference[:1])[0]
    assert reference[0] == bottom
    assert evidence["boundary_reference"] == bottom
    assert evidence["maximum_absolute_residual"] == max(
        abs(np.asarray(readings) - bottom)
    )
    assert evidence["status"] == "unverified_discrepancy"


def test_substantive_fixed_sensor_disagreement_is_separate_unverified_evidence(
    tmp_path,
):
    require_backend()
    path = tmp_path / "numerical_discrepancy"
    _, raw_sensor = save_manufactured(path, fixed_offset=1.0)
    with PosteriorSamples.open(path) as samples:
        report = samples.diagnostics()
        assert report.qualified  # This is chain convergence, not numerical accuracy.
        evidence = report.fixed_sensor_consistency[0]
        assert evidence["sensor"] == 0
        assert evidence["boundary_reference"] == 300.0
        assert evidence["observed_minimum"] == evidence["observed_maximum"] == 301.0
        assert evidence["maximum_absolute_residual"] == 1.0
        assert evidence["status"] == "unverified_discrepancy"
        assert np.array_equal(samples.sensor_predictions(chain=0), raw_sensor[0])
        assert report.per_quantity["sensor_temperature"]["mean_mcse"][0] == 0.0
        assert samples.summary().qualified
    with PosteriorSamples.open(path) as samples:
        import json

        saved = json.loads((path / "diagnostics.json").read_text())
        assert saved["fixed_sensor_consistency"][0] == evidence


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_fixed_reference_interpolates_nonconstant_prescribed_temperature(cell):
    record = make_record(nmesh=3, cell=cell)
    boundary_x = record.arrays["coordinates"][record.arrays["boundary_dofs"], 0]
    # A separate frozen boundary field tests the authoritative interpolation;
    # no saved calculation or posterior identity is reused for sampling here.
    record = replace(
        record, arrays=record.arrays | {"boundary_values": 1 + 2 * boundary_x}
    )
    _, fixed, reference = _physical_observables(record, include_fixed_temperatures=True)
    assert np.array_equal(fixed, [True, False])
    assert np.isclose(reference[0], 1.5, rtol=0, atol=1e-14)
    assert np.isnan(reference[1])
