"""Exact calibration mechanics do not qualify the physical heat study."""

import json
import numpy as np
import pytest

from hmc_validation import (
    StudyConfig,
    sbc_ranks,
    sbc_uniform_envelope,
    _cdf_deviation,
    _rank_selection,
    _study_manifest,
)


def test_rank_counts_draws_below_truth():
    draws = np.array([[1.0, 3.0, 4.0], [0.0, 2.0, 5.0]])
    assert np.array_equal(sbc_ranks(np.array([2.0, 6.0]), draws, seed=8), [1, 3])


def test_exact_ties_are_randomized_including_both_endpoints():
    truth = np.ones(1000)
    ranks = sbc_ranks(truth, np.ones((1000, 3)), seed=8)
    assert set(ranks) == {0, 1, 2, 3}
    assert np.array_equal(ranks, sbc_ranks(truth, np.ones((1000, 3)), seed=8))
    assert np.max(np.abs(np.bincount(ranks, minlength=4) - 250)) < 60


@pytest.mark.parametrize(
    "truth,draws",
    [
        (np.array([np.nan]), [[1.0]]),
        (np.array([1.0]), [[np.inf]]),
        (np.array([1.0, 2.0]), [[1.0, 2.0]]),
    ],
)
def test_rank_rejects_invalid_values(truth, draws):
    with pytest.raises(ValueError):
        sbc_ranks(truth, np.array(draws), seed=8)


def test_exact_scalar_gaussian_posterior_and_deliberate_bias():
    # theta~N(0,1), y|theta~N(theta,1), theta|y~N(y/2,1/2).
    # This verifies rank mechanics using an exact posterior, not heat convergence.
    rng = np.random.default_rng(92)
    theta = rng.normal(size=2000)
    data = theta + rng.normal(size=2000)
    draws = data[:, None] / 2 + rng.normal(size=(2000, 100)) / np.sqrt(2)
    ranks = sbc_ranks(theta, draws, seed=8)
    biased = sbc_ranks(theta, draws + 2.0, seed=8)
    assert np.max(np.abs(_cdf_deviation(ranks, posterior_count=100))) < 0.05
    assert np.max(np.abs(_cdf_deviation(biased, posterior_count=100))) > 0.5


def test_discrete_uniform_cdf_uses_inclusive_rank_probability():
    ranks = np.array([0, 1, 2, 3])
    assert np.array_equal(_cdf_deviation(ranks, posterior_count=3), np.zeros(4))
    assert np.allclose(
        _cdf_deviation(np.array([0, 0]), posterior_count=3), [0.75, 0.5, 0.25, 0]
    )


def test_envelope_simulates_simultaneous_cdf_and_bonferroni():
    result = sbc_uniform_envelope(replicates=4, posterior_count=3, quantities=5, seed=9)
    rng = np.random.default_rng(9)
    ranks = rng.integers(0, 4, size=(100000, 4))
    cdf = (ranks[:, :, None] <= np.arange(4)).mean(axis=1)
    maxima = np.max(np.abs(cdf - np.arange(1, 5) / 4), axis=1)
    expected = np.quantile(maxima, 0.99, method="higher")
    assert result["simulations"] == 100000
    assert result["alpha_per_quantity"] == 0.01
    assert result["critical_deviation"] == expected
    assert np.allclose(result["lower"], np.maximum(0, np.arange(1, 5) / 4 - expected))
    assert np.allclose(result["upper"], np.minimum(1, np.arange(1, 5) / 4 + expected))


def test_reference_experiment_has_fixed_physical_observations():
    config = StudyConfig.from_json("validation/hmc_reference.json")
    assert config.sensor_points.shape == (256, 2)
    assert np.all((config.sensor_points > 0) & (config.sensor_points < 1))
    assert config.sigma == 1e-3
    assert config.inference_meshes == (16, 32, 64)
    assert config.mu == np.log(2)
    assert (config.kappa, config.tau, config.h_ref) == (2, 1, 1)
    assert config.seed == 20261026
    assert config.cell_type == "quadrilateral"
    assert not config.sensor_points.flags.writeable


def test_rank_selection_balances_chains_and_reports_double_interval():
    import os

    try:
        import arviz_stats
    except ImportError:
        if os.environ.get("S2DHI_HMC_QUALIFICATION") == "1":
            raise
        pytest.skip("optional diagnostics extra is absent")
    assert arviz_stats.__version__ == "1.3.3"
    rng = np.random.default_rng(22)
    values = rng.normal(size=(4, 1200, 5))
    selected = _rank_selection(values)
    assert np.asarray(selected["indices"]).shape == (4, 25)
    assert np.asarray(selected["double_indices"]).shape == (4, 25)
    assert len(selected["quantiles"]) == 19
    assert np.asarray(selected["quantile_ess"]).shape == (5, 19)
    assert selected["interval"] >= 1
    assert np.all(
        np.diff(selected["double_indices"], axis=1) == 2 * selected["interval"]
    )
    assert "residual_autocorrelation" in selected


def test_study_resume_binds_configuration_numerics_and_kind(tmp_path):
    config = StudyConfig.from_json("validation/hmc_reference.json")
    first = _study_manifest(
        config, tmp_path / "study", kind="pilot", extra={"data_hash": "a"}
    )
    assert (
        _study_manifest(
            config, tmp_path / "study", kind="pilot", extra={"data_hash": "a"}
        )
        == first
    )
    with pytest.raises(ValueError, match="different|changed|identity"):
        _study_manifest(
            config, tmp_path / "study", kind="pilot", extra={"data_hash": "b"}
        )
    with pytest.raises(ValueError, match="different|changed|identity"):
        _study_manifest(
            config, tmp_path / "study", kind="sbc", extra={"data_hash": "a"}
        )


def test_analytic_observation_record_is_fixed_and_reused(tmp_path):
    from hmc_validation import _observations

    config = StudyConfig()
    points, values, sigma = _observations(config)
    expected = 300 + (points[:, 1] - 0.5 * points[:, 1] ** 2) / 2
    expected += np.random.default_rng(20261024).normal(0, 1e-3, 256)
    assert np.array_equal(values, expected)
    path = tmp_path / "observations.npz"
    np.savez(path, points=points, values=values, sigma=sigma)
    actual = _observations(config, path)
    assert all(np.array_equal(a, b) for a, b in zip(actual, (points, values, sigma)))


def test_antithetic_indicator_ess_recomputed_after_prethinning(monkeypatch):
    import hmc_validation as study

    seen = []

    def ess(values):
        seen.append(values.shape[1])
        return np.full((5, 19), 9000.0 if len(seen) == 1 else 1000.0)

    monkeypatch.setattr(study, "_indicator_ess", ess)
    values = np.random.default_rng(8).normal(size=(4, 2000, 5))
    selection = study._rank_selection(values)
    assert seen == [2000, 1000]
    assert selection["antithetic_prethin"] == 2
    assert selection["interval"] == 8  # 2*ceil(4*1000/1000)
    assert selection["enough"]


def test_rank_selection_requires_enough_for_doubled_interval(monkeypatch):
    import hmc_validation as study

    monkeypatch.setattr(study, "_indicator_ess", lambda values: np.full((5, 19), 40.0))
    selection = study._rank_selection(np.random.default_rng(8).normal(size=(4, 100, 5)))
    assert selection["interval"] == 10
    assert selection["required_draws_per_chain"] == 481
    assert not selection["enough"]
    assert "indices" not in selection


def test_only_finite_insufficient_evidence_drives_chain_extension():
    from hmc_validation import _extension_target
    from types import SimpleNamespace

    quantity = {"analytically_fixed": [False], "bulk_ess": [200.0], "tail_ess": [100.0]}
    report = SimpleNamespace(
        divergences=0, per_quantity={"h": quantity}, reasons=["low ESS"]
    )
    assert _extension_target(report, 2000) == 16000
    report.divergences = 1
    with pytest.raises(RuntimeError, match="divergences"):
        _extension_target(report, 2000)
    report.divergences = 0
    quantity["tail_ess"] = [np.nan]
    with pytest.raises(RuntimeError, match="nonfinite"):
        _extension_target(report, 2000)
    quantity["tail_ess"] = [500.0]
    report.reasons = ["h: nonfinite mean_mcse"]
    with pytest.raises(RuntimeError, match="nonfinite"):
        _extension_target(report, 2000)


def test_study_resume_rejects_changed_numerical_identity(tmp_path, monkeypatch):
    import posterior_store

    monkeypatch.setattr(
        posterior_store, "numerical_environment", lambda: {"native": "first"}
    )
    config = StudyConfig()
    _study_manifest(config, tmp_path / "study", kind="sbc", extra={"replicates": 100})
    monkeypatch.setattr(
        posterior_store, "numerical_environment", lambda: {"native": "second"}
    )
    with pytest.raises(ValueError, match="changed"):
        _study_manifest(
            config, tmp_path / "study", kind="sbc", extra={"replicates": 100}
        )


def test_failed_replication_stays_visible_without_replacing_prior_truth(
    tmp_path, monkeypatch
):
    import hmc_validation as study

    # Exercise the orchestration branch that preserves a previously failed draw.
    # It must never call a heat model or sampler for that failed experiment.
    monkeypatch.setattr(
        study,
        "_qualified_pilot",
        lambda config: (
            {"meshes": {"16": {"tolerances": [1e-10, 1e-12]}}, "selected_spread": 0.3},
            "pilot-hash",
        ),
    )
    monkeypatch.setattr(study, "_require_native", lambda manifest: None)
    import posterior_store

    monkeypatch.setattr(
        posterior_store, "numerical_environment", lambda: {"test": "mechanics"}
    )
    config = StudyConfig()
    output = tmp_path / "sbc"
    study._study_manifest(
        config,
        output,
        kind="sbc",
        extra={"replicates": 1, "pilot_qualification_sha256": "pilot-hash"},
    )
    directory = output / "replicate-000"
    directory.mkdir()
    failure = {
        "exception": "HeatSolveFailure",
        "message": "finite solve failed",
        "seed": 32,
    }
    (directory / "failure.json").write_text(json.dumps(failure))
    from hmc_sbc_worker import issue_attempt, atomic_json

    manifest = json.loads((output / "manifest.json").read_text())
    issued = issue_attempt(
        output,
        index=0,
        replicates=1,
        identity=manifest["identity_sha256"],
        session="preserved",
        command=["fixture"],
    )
    issue = json.loads(issued.read_text())
    atomic_json(
        directory / "result.json",
        {
            "attempt": issue["attempt"],
            "replicate": 0,
            "replicates": 1,
            "identity": manifest["identity_sha256"],
            "status": "failed",
            "seeds": study._sbc_seeds(config, 0),
            "failure": failure,
        },
    )
    result = study.run_sbc(config, output, replicates=1)
    assert result["qualified"] == 0
    assert not result["calibration_complete"]
    assert result["records"][0]["failure"] == failure
    assert not (directory / "generated.npz").exists()


def test_saved_field_artifact_recovers_interrupted_rename_and_detects_change(tmp_path):
    from hmc_validation import _save_arrays, _array_record

    arrays = tmp_path / "data.npz"
    record = tmp_path / "data.json"
    info = _save_arrays(arrays, record, {"truth": np.array([2.0])}, {"seed": 8})
    temporary = arrays.with_suffix(".tmp.npz")
    arrays.rename(temporary)
    assert _array_record(arrays, record) == info
    assert arrays.exists() and not temporary.exists()
    np.savez(arrays, truth=np.array([3.0]))
    with pytest.raises(ValueError, match="changed"):
        _array_record(arrays, record)


def test_physical_rank_quantities_use_area_mean_and_fixed_nodes(monkeypatch):
    from types import SimpleNamespace
    import posterior_diagnostics
    from hmc_validation import _quantities, RANK_POINTS

    record = SimpleNamespace(
        arrays={"coordinates": np.array([*RANK_POINTS, (0.5, 0.5)])}
    )
    weights = np.array(
        [
            [0.1, 0.2, 0.3, 0.1, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(
        posterior_diagnostics, "_physical_observables", lambda record: (weights, None)
    )
    fields = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0]])
    actual = _quantities(record, fields)
    assert np.allclose(actual, [[3.3, 1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0, 2.0]])


def test_sampling_cost_matches_relative_and_absolute_run_locations(
    tmp_path, monkeypatch
):
    from hmc_validation import _event, _sampling_seconds

    monkeypatch.chdir(tmp_path)
    _event(
        tmp_path, "sampling_segment_elapsed", run=str(tmp_path / "run"), seconds=1.25
    )
    _event(tmp_path, "sampling_segment_elapsed", run="run", seconds=2.75)
    assert _sampling_seconds(tmp_path, "run") == 4.0
    assert _sampling_seconds(tmp_path, tmp_path / "run") == 4.0


def test_small_heat_study_paths_preserve_actual_chain_evidence_and_resume(tmp_path):
    """Toy call-path check only: n2, sigma.05 and 4warmup/6draws cannot qualify."""
    import os

    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip("optional full-field sampling qualification")
    from hmc_blackjax import require_sampling_environment

    require_sampling_environment()
    from hmc_validation import (
        _observations,
        _problem,
        _force_comparison,
        _sample,
        _chain_evidence,
        _sampling_seconds,
        _encoded,
    )

    config = StudyConfig(
        sigma=0.05, initial_step_size=0.01, step_size_bounds=(1e-6, 0.02)
    )
    observations = _observations(config)
    output = tmp_path / "toy-study"
    output.mkdir()
    path = output / "posterior"
    with _problem(config, 2, observations, (1e-10, 1e-12)) as problem:
        initial = problem.initial_state(2.0)
        center = initial.coordinates["z"]
        force = _force_comparison(
            config,
            2,
            observations,
            [center, center + np.linspace(-0.05, 0.05, len(center))],
            ["median", "perturbation"],
            output,
        )
        assert len(force["comparisons"]) == 6
        assert force["passing_tolerance_indices"]
        assert force["reference_cost"]["solve_counts"]["heat"] == 2
        assert force["reference_cost"]["solve_counts"]["adjoint"] == 2
        assert force["reference_cost"]["seconds"] > 0
        assert len(force["candidate_costs"]) == 3
        with _sample(
            config,
            problem,
            initial,
            path,
            output,
            seed=817,
            spread=0.1,
            warmup=4,
            draws=6,
        ) as samples:
            before = samples.load_draws()
            assert before.shape == (4, 6, 9)
            assert np.isfinite(before).all() and (before > 0).all()
            assert not samples.diagnostics().qualified
        evidence = _chain_evidence(path, config.step_size_bounds[0])
        assert not evidence["passes_spread_criteria"]
        assert evidence["seed"] == 817
        assert len(evidence["random_streams"]) == 4
        assert (
            len({_encoded(s["initialization"]) for s in evidence["random_streams"]})
            == 4
        )
        for chain in evidence["chains"]:
            assert chain["warmup_count"] == 4
            assert chain["retained_count"] == 6
            assert np.isfinite(chain["initial_phi"]) and chain["initial_phi"] >= 0
            assert (
                chain["solve_counts"]["heat"] > 0
                and chain["solve_counts"]["adjoint"] > 0
            )
            assert 0 <= chain["mean_retained_acceptance"] <= 1
            for row in chain["statistics"]["retained"]:
                assert {
                    "delta_energy",
                    "duration",
                    "actual_duration",
                    "step_size",
                    "acceptance_probability",
                    "num_steps",
                    "divergent",
                } <= row.keys()
                assert np.pi / 4 <= row["duration"] <= 3 * np.pi / 4
                assert np.isclose(
                    row["actual_duration"], row["num_steps"] * row["step_size"]
                )
        elapsed = _sampling_seconds(output, path)
        assert elapsed > 0
        with _sample(
            config,
            problem,
            initial,
            path,
            output,
            seed=817,
            spread=0.1,
            warmup=4,
            draws=6,
        ) as resumed:
            assert np.array_equal(resumed.load_draws(), before)
            assert not resumed.diagnostics().qualified
        after = _chain_evidence(path, config.step_size_bounds[0])
        assert _encoded(after) == _encoded(evidence)
        assert _sampling_seconds(output, path) >= elapsed


def test_worker_scientific_source_is_bound_to_study_identity(tmp_path, monkeypatch):
    import hmc_validation as study
    import posterior_store

    monkeypatch.setattr(
        posterior_store, "numerical_environment", lambda: {"test": "mechanics"}
    )
    manifest = study._study_manifest(
        StudyConfig(), tmp_path / "study", kind="sbc", extra={"replicates": 2}
    )
    assert "hmc_sbc_worker" in manifest["identity"]["study_sources"]
    original_hash = study._file_hash
    monkeypatch.setattr(
        study,
        "_file_hash",
        lambda path: (
            "changed-worker"
            if str(path).endswith("hmc_sbc_worker.py")
            else original_hash(path)
        ),
    )
    with pytest.raises(ValueError, match="Study identity changed"):
        study._study_manifest(
            StudyConfig(), tmp_path / "study", kind="sbc", extra={"replicates": 2}
        )


def test_extension_keeps_largest_saved_draw_target(tmp_path, monkeypatch):
    import hmc_validation as study
    import posterior_samples
    from types import SimpleNamespace

    path = tmp_path / "posterior"
    path.mkdir()

    class Saved:
        metadata = {"total_draws": 8192}
        _store = SimpleNamespace(failures=lambda: [])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        posterior_samples.PosteriorSamples, "open", lambda path: Saved()
    )
    calls = []

    def resume(path, *, total_draws, progress):
        calls.append(total_draws)
        return SimpleNamespace(metadata={"total_draws": total_draws})

    study._sample(
        StudyConfig(),
        SimpleNamespace(resume=resume),
        None,
        path,
        tmp_path,
        seed=1,
        spread=1.0,
        warmup=1000,
        draws=2000,
    )
    assert calls == [8192]


def test_interface_coverage_reports_regions_separately():
    from hmc_validation import interface_coverage

    xy = np.array([[0.62, 0.0], [0.1, 0.0], [0.9, 0.0]])
    report = interface_coverage(xy, [0.5, 0.5, 2.5], [1.5, 1.5, 3.5], [3.0, 1.0, 3.0])
    assert report["near_count"] == 1
    assert report["near_coverage"] == 0.0
    assert report["away_count"] == 2
    assert report["away_coverage"] == 1.0


def test_interface_coverage_empty_region_is_unresolved():
    from hmc_validation import interface_coverage

    report = interface_coverage([[0.1, 0.0]], [0.5], [1.5], [1.0])
    assert report["near_coverage"] is None
    with pytest.raises(ValueError):
        interface_coverage([[0.1, 0.0]], [2.0], [1.0], [1.0])


def test_fine_truth_refines_before_one_frozen_noise_draw(tmp_path, monkeypatch):
    mesh = _mesh_mechanics_only(monkeypatch)

    calls = []
    config = StudyConfig()

    def temperatures(config, truth, nmesh):
        calls.append(nmesh)
        # First comparison fails; the next is below 0.1 sigma.
        offset = {128: 0.001, 256: 0.00001, 512: 0.0}[nmesh]
        return np.full(256, 300.0 + offset)

    monkeypatch.setattr(mesh, "_truth_temperatures", temperatures)
    monkeypatch.setattr(mesh.hv, "_require_native", lambda manifest: None)
    path = mesh.make_truth_data(config, truth="smooth", output=tmp_path / "truth")
    assert calls == [128, 256, 512]
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["truth_mesh"] == 512
    assert len(record["refinement"]) == 2
    assert record["refinement"][-1]["maximum_change_over_sigma"] < 0.1
    with np.load(path) as archive:
        expected = 300.0 + np.random.default_rng(20261026).normal(0, config.sigma, 256)
        assert np.array_equal(archive["values"], expected)
        assert np.array_equal(archive["noiseless"], np.full(256, 300.0))
    original = path.read_bytes()
    assert mesh.make_truth_data(config, truth="smooth", output=path.parent) == path
    assert path.read_bytes() == original
    assert calls == [128, 256, 512]
    # Each mesh reads the same bytes; the physical prior has no mesh-dependent scale.
    for nmesh in config.inference_meshes:
        observation, info = mesh._truth_observations(config, path)
        assert info["array_sha256"] == record["array_sha256"]
        assert observation[1].tobytes() == expected.tobytes()
        assert observation[0].tobytes() == config.sensor_points.tobytes()
        assert observation[2].tobytes() == np.asarray(config.sigma).tobytes()
        assert mesh._prior_case(config, nmesh) == "baseline"


def test_truth_data_detects_changed_bytes(tmp_path, monkeypatch):
    mesh = _mesh_mechanics_only(monkeypatch)

    monkeypatch.setattr(mesh, "_truth_temperatures", lambda *a: np.full(256, 300.0))
    monkeypatch.setattr(mesh.hv, "_require_native", lambda manifest: None)
    path = mesh.make_truth_data(StudyConfig(), truth="sharp", output=tmp_path / "truth")
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        mesh._truth_observations(StudyConfig(), path)


def test_prior_sensitivity_is_predeclared_and_one_at_a_time():
    from dataclasses import replace
    from hmc_mesh_study import _prior_case

    config = StudyConfig()
    for key, value in (("kappa", 1.0), ("kappa", 4.0), ("tau", 0.5), ("tau", 2.0)):
        changed = replace(config, **{key: value})
        assert _prior_case(changed, 16) != "baseline"
        with pytest.raises(ValueError):
            _prior_case(changed, 32)
    with pytest.raises(ValueError):
        _prior_case(replace(config, kappa=1.0, tau=0.5), 16)


def test_contiguous_block_uncertainty_is_seeded_and_refuses_short_chains():
    from hmc_mesh_study import _block_uncertainty

    rng = np.random.default_rng(8)
    values = rng.normal(size=(4, 256, 2))
    acceptance = rng.uniform(0.7, 1.0, (4, 256))
    costs = np.ones((4, 256))
    # An independent analytical ESS stand-in isolates resampling mechanics.
    ess = lambda x: np.full(x.shape[-1], x.shape[0] * x.shape[1], dtype=float)
    result = _block_uncertainty(
        values, acceptance, costs, costs, correlation_time=3.0, seed=2, ess=ess
    )
    assert result["resamples"] == 500
    assert result["block_length"] >= 6
    assert result["status"] == "resolved"
    assert result == _block_uncertainty(
        values, acceptance, costs, costs, correlation_time=3.0, seed=2, ess=ess
    )
    assert result["acceptance_interval"][0] < result["acceptance_interval"][1]
    short = _block_uncertainty(
        values[:, :10],
        acceptance[:, :10],
        costs[:, :10],
        costs[:, :10],
        correlation_time=3.0,
        seed=2,
        ess=ess,
    )
    assert short["status"] == "unresolved"


def test_predictive_residuals_include_measurement_noise():
    from hmc_mesh_study import _predictive_summary

    result = _predictive_summary(
        np.full((4, 100, 2), 300.0),
        np.array([300.0, 300.001]),
        np.asarray(0.001),
        seed=1,
    )
    assert np.allclose(result["predictive_sd"], 0.001)
    assert np.allclose(result["standardized_residual"], [0.0, -1.0])
    assert np.all(np.asarray(result["upper"]) > np.asarray(result["lower"]))


def test_missing_mesh_report_lists_all_fourteen_outcomes(tmp_path):
    from hmc_validation import write_study_report

    path = tmp_path / "report.md"
    write_study_report([], path)
    report = path.read_text()
    assert report.count("| missing |") == 14
    assert "spatial diagnostic" in report
    assert "not complete" in report


def _mesh_mechanics_only(monkeypatch):
    """Replace native provenance only for array/record mechanics, never physics."""
    import hmc_mesh_study as mesh
    from dataclasses import asdict

    def manifest(config, output, *, kind, extra):
        from pathlib import Path

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        identity = dict(kind=kind, config=asdict(config), extra=extra)
        record = dict(identity=identity, identity_sha256=mesh.hv._hash(identity))
        mesh.hv._write(output / "manifest.json", record)
        return record

    monkeypatch.setattr(mesh.hv, "_study_manifest", manifest)
    monkeypatch.setattr(mesh.hv, "_require_native", lambda manifest: None)
    monkeypatch.setattr(mesh, "_truth_temperatures", lambda *args: np.full(256, 300.0))
    return mesh


def test_actual_truth_pilot_mismatch_is_preserved_before_sampling(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    mesh = _mesh_mechanics_only(monkeypatch)
    config = replace(StudyConfig(), pilot_output=str(tmp_path / "pilot"))
    path = mesh.make_truth_data(config, truth="smooth", output=tmp_path / "truth")
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    mesh.hv._write(
        pilot / "manifest.json",
        {"identity": {"extra": {"observation_sha256": "wrong-data"}}},
    )
    monkeypatch.setattr(mesh.hv, "_qualified_pilot", lambda c: ({}, "pilot-hash"))
    monkeypatch.setattr(
        mesh.hv,
        "_problem",
        lambda *args: pytest.fail("Mismatched data must never launch a posterior."),
    )
    result = mesh.run_mesh_study(
        config, data_path=path, nmesh=16, output=tmp_path / "mesh"
    )
    assert result["status"] == "failed"
    assert "do not match" in result["failure"]["message"]
    assert (tmp_path / "mesh" / "failure.json").exists()
    report = tmp_path / "report.md"
    mesh.write_study_report([tmp_path / "mesh" / "manifest.json"], report)
    assert "| failed |" in report.read_text()
    assert report.read_text().count("| missing |") == 13
    with pytest.raises(RuntimeError, match="Preserved"):
        mesh.run_mesh_study(config, data_path=path, nmesh=16, output=tmp_path / "mesh")
    saved = tmp_path / "mesh" / "result.json"
    saved.write_text(saved.read_text().replace('"failed"', '"sampled"'))
    with pytest.raises(ValueError, match="checksum"):
        mesh.write_study_report([tmp_path / "mesh" / "manifest.json"], report)


def test_refinement_evidence_cannot_claim_another_truth_mesh(tmp_path, monkeypatch):
    mesh = _mesh_mechanics_only(monkeypatch)
    path = mesh.make_truth_data(StudyConfig(), truth="sharp", output=tmp_path / "truth")
    record = json.loads(path.with_suffix(".json").read_text())
    record["truth_mesh"] = 128
    path.with_suffix(".json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="refinement|checksum"):
        mesh._truth_observations(StudyConfig(), path)


def test_block_uncertainty_recomputes_ess_for_every_paired_resample():
    from hmc_mesh_study import _block_uncertainty

    rng = np.random.default_rng(17)
    values = rng.normal(size=(4, 128, 1))
    calls = []

    def measured_ess(sample):
        calls.append(sample.copy())
        return np.array([sample.size / (1 + abs(sample.mean()))])

    costs = np.exp(values[..., 0])
    result = _block_uncertainty(
        values,
        np.full((4, 128), 0.8),
        costs,
        2 * costs,
        correlation_time=2,
        seed=19,
        ess=measured_ess,
    )
    assert len(calls) == 500
    assert not np.array_equal(calls[0], calls[1])
    assert (
        result["ess_per_second_intervals"][0][0]
        < result["ess_per_second_intervals"][0][1]
    )
    assert np.allclose(
        np.asarray(result["ess_per_second_intervals"]) / 2,
        result["ess_per_solve_intervals"],
    )
    # Within each sampled contiguous block, the original chain's neighboring
    # states remain consecutive; no synthetic transitions inside a block.
    for chain in range(4):
        for start in range(0, 128, result["block_length"]):
            block = calls[0][chain, start : start + result["block_length"], 0]
            original = np.flatnonzero(values[chain, :, 0] == block[0])[0]
            assert np.array_equal(
                block, values[chain, original : original + len(block), 0]
            )


def test_physical_report_keeps_sensor_worst_case_and_cost_labels(tmp_path, monkeypatch):
    """Synthetic chains exercise summary assembly, never qualify heat physics."""
    from types import SimpleNamespace
    import hmc_mesh_study as mesh
    import posterior_diagnostics as diagnostics
    from posterior_diagnostics import DiagnosticReport

    rng = np.random.default_rng(23)
    fields = np.exp(rng.normal(size=(4, 256, 4)))
    temperatures = 300 + rng.normal(size=(4, 256, 2)) * 0.001
    xy = np.asarray(mesh.hv.RANK_POINTS)
    weights = np.array([[0.25] * 4, [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])
    monkeypatch.setattr(
        diagnostics,
        "_physical_observables",
        lambda record: (weights, np.zeros(2, dtype=bool)),
    )
    monkeypatch.setattr(mesh, "_ess", lambda values: np.full(values.shape[-1], 500.0))
    rows = [
        [
            SimpleNamespace(
                h=fields[c, d],
                sensor_temperature=temperatures[c, d],
                statistics=dict(
                    acceptance_probability=0.7 + 0.3 * rng.random(),
                    delta_energy=rng.normal(),
                    trajectory_seconds=1.0,
                    solve_counts={"heat": 3, "adjoint": 2},
                ),
            )
            for d in range(256)
        ]
        for c in range(4)
    ]
    record = SimpleNamespace(
        arrays=dict(
            coordinates=xy,
            sensor_values=np.full(2, 300.0),
            sensor_sigma=np.asarray(0.001),
            sensor_points=xy[:2],
        )
    )
    summary = SimpleNamespace(
        coordinates=xy,
        cells=np.array([[0, 1, 2, 3]]),
        mean=fields.mean(axis=(0, 1)),
        quantiles=np.quantile(fields, [0.05, 0.5, 0.95], axis=(0, 1)),
    )
    store = SimpleNamespace(
        records=lambda c: iter(rows[c]),
        checkpoint=lambda c: SimpleNamespace(counters={"heat": 800, "adjoint": 600}),
    )
    samples = SimpleNamespace(
        metadata={"chains": 4},
        _counts=lambda: np.full(4, 256),
        _record=record,
        _store=store,
        summary=lambda: summary,
        coordinates=xy,
        output=tmp_path / "posterior",
    )

    def stats(locations, ess):
        count = len(locations)
        return dict(
            bulk_ess=[ess] * count,
            tail_ess=[ess] * count,
            analytically_fixed=[False] * count,
            locations=locations,
            mean_mcse=[0.01] * count,
            quantile_mcse=[[0.02] * count] * 3,
        )

    report = DiagnosticReport(
        status="qualified",
        qualified=True,
        per_quantity=dict(
            conductivity=stats(xy.tolist(), 800),
            area_mean=stats([None], 800),
            left_minus_right=stats([None], 800),
            bottom_minus_top=stats([None], 800),
            sensor_temperature=stats(xy[:2].tolist(), 500),
        ),
        worst_locations={},
        divergences=0,
        counts=[256] * 4,
        package_version="mechanics",
        reasons=[],
        probabilities=[0.05, 0.5, 0.95],
        common_draws=256,
        requested_total=256,
        identity={},
        settings={},
    )
    result = mesh._physical_evidence(StudyConfig(), samples, report, "sharp", tmp_path)
    assert len(result["cost"]["labels"]) == len(result["cost"]["ess"]) == 7
    assert "sensor_temperature" in result["cost"]["labels"][-1]
    assert result["shared_locations"][-1]["location"] == xy[0].tolist()
    assert result["cost"]["retained_seconds"] == 1024
    assert result["cost"]["retained_heat_adjoint_solves"] == 5120
    assert len(result["predictive"]["mean"]) == 2
    assert result["predictive"]["includes_observation_noise"]
    assert result["uncertainty"]["status"] == "resolved"


def test_small_mesh_truth_temperature_paths():
    """Tiny native solves check both declared truth paths, never fine refinement."""
    import os

    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip("optional native heat study mechanics")
    from hmc_mesh_study import _truth_temperatures

    for truth in ("smooth", "sharp"):
        for nmesh in (2, 4):
            readings = _truth_temperatures(StudyConfig(), truth, nmesh)
            assert readings.shape == (256,)
            assert np.isfinite(readings).all()
            assert np.all(readings > 300.0)
            assert np.all(readings < 302.0)


def test_efficiency_comparisons_use_independent_resampling_streams():
    from hmc_mesh_study import _comparison_uncertainty

    first = dict(
        status="resolved",
        seed=10,
        acceptance_resamples=[0.7] * 500,
        ess_per_second_resamples=[[2.0] * 7] * 500,
        ess_per_solve_resamples=[[4.0] * 7] * 500,
    )
    second = dict(
        status="resolved",
        seed=11,
        acceptance_resamples=[0.8] * 500,
        ess_per_second_resamples=[[6.0] * 7] * 500,
        ess_per_solve_resamples=[[2.0] * 7] * 500,
    )
    result = _comparison_uncertainty(first, second)
    assert np.allclose(result["acceptance_difference_interval"], [0.1, 0.1])
    assert np.allclose(result["ess_per_second_ratio_intervals"], np.full((5, 2), 3.0))
    assert np.allclose(result["ess_per_solve_ratio_intervals"], np.full((5, 2), 0.5))
    assert _comparison_uncertainty(first, first)["status"] == "unresolved"
