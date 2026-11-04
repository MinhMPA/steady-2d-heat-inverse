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
