"""Failure, continuation and provenance evidence for physical comparisons."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import builtins
import json
import numpy as np
import pytest

import hmc_mesh_study as mesh
from tests.test_hmc_validation import _mesh_mechanics_only


def test_abandoned_heat_work_enters_total_but_not_retained_cost():
    sample = SimpleNamespace(
        metadata={"chains": 2},
        _store=SimpleNamespace(
            checkpoint=lambda chain: SimpleNamespace(
                counters={"heat": 10, "adjoint": 7}
            ),
            failures=lambda: iter(
                [
                    {"aborted_work": {"heat": 3, "adjoint": 2}},
                    {"aborted_work": {"heat": 1, "adjoint": 0}},
                ]
            ),
        ),
    )
    counts = mesh._solve_totals(sample)
    assert counts["committed"] == {"heat": 20, "adjoint": 14}
    assert counts["abandoned"] == {"heat": 4, "adjoint": 2}
    assert counts["total"] == {"heat": 24, "adjoint": 16}


def test_result_publication_recovers_interruption_between_result_and_digest(
    tmp_path, monkeypatch
):
    path = tmp_path / "result.json"
    original = mesh.hv._write

    def interrupted(destination, value):
        if Path(destination).name == "result-sha256.json":
            raise KeyboardInterrupt("after result before digest")
        original(destination, value)

    monkeypatch.setattr(mesh.hv, "_write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        mesh._publish_record(path, {"status": "sampled", "generation": 0})
    monkeypatch.setattr(mesh.hv, "_write", original)
    assert mesh._load_publication(path) == {"status": "sampled", "generation": 0}
    path.write_text('{"status":"changed"}')
    with pytest.raises(ValueError, match="checksum"):
        mesh._load_publication(path)


def test_fine_truth_passing_refinement_metadata_is_immutable(tmp_path, monkeypatch):
    study = _mesh_mechanics_only(monkeypatch)
    path = study.make_truth_data(
        study.hv.StudyConfig(), truth="smooth", output=tmp_path / "truth"
    )
    record_path = path.with_suffix(".json")
    record = json.loads(record_path.read_text())
    record["refinement"][-1]["maximum_change_over_sigma"] = 0.01
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        study._truth_observations(study.hv.StudyConfig(), path)


def test_fine_truth_refinement_archive_is_immutable(tmp_path, monkeypatch):
    study = _mesh_mechanics_only(monkeypatch)
    path = study.make_truth_data(
        study.hv.StudyConfig(), truth="sharp", output=tmp_path / "truth"
    )
    refinement = path.parent / "refinement.json"
    refinement.write_text(
        '[{"coarse":128,"fine":256,"maximum_change_over_sigma":0.01}]'
    )
    with pytest.raises(ValueError, match="refinement"):
        study._truth_observations(study.hv.StudyConfig(), path)


def test_explicit_extension_preserves_generation_and_recovers_request(tmp_path):
    artifact = tmp_path / "posterior" / "chains.sqlite"
    artifact.parent.mkdir()
    artifact.write_bytes(b"original scientific histories")
    previous = dict(
        status="sampled",
        generation=0,
        requested_total=8,
        checksums={"posterior/chains.sqlite": mesh.hv._file_hash(artifact)},
    )
    mesh._publish_record(tmp_path / "result.json", previous)
    request, cached = mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
    assert cached is None
    assert request["generation"] == 1 and request["total_draws"] == 16
    archive = tmp_path / "history" / "generation-0000"
    assert mesh._load_publication(archive / "result.json") == previous
    artifact.write_bytes(b"extended scientific histories")
    assert (
        archive / "posterior" / "chains.sqlite"
    ).read_bytes() == b"original scientific histories"
    resumed, cached = mesh._prepare_analysis(
        tmp_path, total_draws=None, reanalyze=False
    )
    assert resumed == request and cached is None
    with pytest.raises(ValueError, match="request"):
        mesh._prepare_analysis(tmp_path, total_draws=32, reanalyze=False)
    following = dict(
        status="sampled",
        generation=1,
        requested_total=16,
        checksums={"posterior/chains.sqlite": mesh.hv._file_hash(artifact)},
    )
    mesh._publish_record(tmp_path / "result.json", following)
    # A crash after publication but before removing the extension request recovers.
    request, cached = mesh._prepare_analysis(
        tmp_path, total_draws=None, reanalyze=False
    )
    assert cached == following and request is None
    with pytest.raises(ValueError, match="larger"):
        mesh._prepare_analysis(tmp_path, total_draws=8, reanalyze=False)


def test_partial_failed_chains_keep_measured_exploration_and_cost(tmp_path):
    @dataclass
    class Report:
        qualified: bool = False
        divergences: int = 1
        worst_locations: dict = None
        per_quantity: dict = None

    report = Report(
        worst_locations={"conductivity": [0.0, 0.0]},
        per_quantity={
            "area_mean": {
                "bulk_ess": [6.0],
                "tail_ess": [5.0],
                "analytically_fixed": [False],
            }
        },
    )
    rows = [
        SimpleNamespace(
            statistics={
                "acceptance_probability": 0.7 + i * 0.01,
                "delta_energy": float(i),
                "trajectory_seconds": 2.0,
                "solve_counts": {"heat": 3, "adjoint": 2},
            },
            h=np.array([1.0 + i * 0.01]),
            sensor_temperature=np.array([300.0]),
        )
        for i in range(8)
    ]
    sample = SimpleNamespace(
        metadata={"chains": 4},
        output=tmp_path / "posterior",
        _counts=lambda: np.full(4, 8),
        diagnostics=lambda: report,
        _store=SimpleNamespace(
            records=lambda chain: iter(rows),
            checkpoint=lambda chain: SimpleNamespace(
                counters={"heat": 30, "adjoint": 20}
            ),
            failures=lambda: iter([{"aborted_work": {"heat": 2, "adjoint": 1}}]),
        ),
    )
    result = mesh._partial_chain_evidence(sample, tmp_path)
    assert result["diagnostics"]["qualified"] is False
    assert result["cost"]["total_heat_adjoint_counts"] == {"heat": 122, "adjoint": 81}
    assert result["cost"]["retained_heat_adjoint_solves"] == 160
    assert result["cost"]["retained_seconds"] == 64
    assert result["efficiency"]["minimum_ess"] == 5.0
    assert result["efficiency"]["ess_per_total_solve"] == pytest.approx(5 / 203)
    assert result["acceptance"]["count"] == 32
    assert len(result["autocorrelation"]["acceptance"]) > 0


def test_rendering_stays_outside_parent_numerical_interpreter(tmp_path, monkeypatch):
    observed = []
    original = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.startswith(("matplotlib", "PIL")):
            raise AssertionError("Plotting must not load native libraries into parent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: observed.append(command)
    )
    mesh._study_figures(
        tmp_path,
        {"predictive": {"points": [[0.5, 0.5]], "standardized_residual": [0.0]}},
    )
    assert len(observed) == 1
    assert "--render-figures" in observed[0]


def test_scientific_failure_publishes_available_costs_and_report(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace

    study = _mesh_mechanics_only(monkeypatch)
    config = replace(study.hv.StudyConfig(), pilot_output=str(tmp_path / "pilot"))
    data = study.make_truth_data(config, truth="smooth", output=tmp_path / "truth")
    observations = study.hv._observations(config, data)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    study.hv._write(
        pilot / "manifest.json",
        {
            "identity": {
                "extra": {
                    "observation_sha256": study.hv._hash(
                        [a.tolist() for a in observations]
                    )
                }
            }
        },
    )
    np.savez(
        pilot / "observations.npz",
        points=observations[0],
        values=observations[1],
        sigma=observations[2],
    )
    monkeypatch.setattr(
        study.hv,
        "_qualified_pilot",
        lambda c: (
            {
                "meshes": {"16": {"tolerances": [1e-10, 1e-12], "spread": 0.1}},
                "initial_production_draws": 8,
            },
            "pilot-hash",
        ),
    )
    monkeypatch.setattr(study.hv, "_problem", lambda *args: nullcontext(None))
    monkeypatch.setattr(study.hv, "_optimize", lambda *args: (None, {"success": True}))

    def divergence(*args, **kwargs):
        raise study.hv._ScientificFailure("retained divergence")

    monkeypatch.setattr(study.hv, "_converged_samples", divergence)
    available = dict(
        status="available_unqualified_evidence",
        cost={"total_heat_adjoint_counts": {"heat": 123, "adjoint": 80}},
        efficiency={"minimum_ess": 5.0},
        acceptance={"mean": 0.6},
        energy_error={"mean": 1.0},
        autocorrelation={},
    )
    monkeypatch.setattr(study, "_failure_chain_evidence", lambda output: available)
    result = study.run_mesh_study(
        config, data_path=data, nmesh=16, output=tmp_path / "mesh"
    )
    assert result["status"] == "failed"
    assert result["available_evidence"] == available
    destination = tmp_path / "report.md"
    study.write_study_report([tmp_path / "mesh" / "manifest.json"], destination)
    report = destination.read_text()
    assert '"heat": 123' in report and '"minimum_ess": 5.0' in report
    assert "retained divergence" in report


def test_two_native_physical_reports_leave_library_identity_unchanged(tmp_path):
    """Actual short heat chains and plots; no convergence or fine-truth claim."""
    import os

    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip("optional native physical-summary integration")
    from tests._hmc_helpers import make_problem, _release_problem
    from conductivity_prior import GaussianLogConductivity
    from heat_inference import HeatInverseProblem
    from temperature_observations import TemperatureObservations
    from hmc_blackjax import HMCControls, require_sampling_environment
    from native_provenance import native_build_identity

    require_sampling_environment()
    config = mesh.hv.StudyConfig(
        mu=0.0, sigma=0.05, initial_step_size=0.05, step_size_bounds=(1e-5, 0.2)
    )
    controls = HMCControls(
        initial_step_size=0.05,
        step_size_bounds=(1e-5, 0.2),
        trajectory_interval=(0.5, 1.0),
        initial_spread=0.1,
    )
    for index, truth in enumerate(("smooth", "sharp")):
        output = tmp_path / truth
        output.mkdir()
        base = make_problem(nmesh=4, cell="quadrilateral")
        problem = None
        try:
            record = base.to_record()
            _, temperatures = base.posterior.predict_h(
                mesh._truth_values(truth, record.arrays["coordinates"])
            )
            problem = HeatInverseProblem(
                forward=base.forward,
                observations=TemperatureObservations(
                    base.posterior.observations.points, temperatures, 0.05
                ),
                prior=GaussianLogConductivity(mu=0.0, kappa=2.0, tau=1.0, h_ref=1.0),
            )
            with problem.sample(
                initial=1.0,
                seed=20261114 + index,
                chains=4,
                warmup=8,
                draws=16,
                controls=controls,
                output=output / "posterior",
            ) as samples:
                report = samples.diagnostics()
                assert not report.qualified
                before_chains = mesh.hv._checksums(samples.output)
                evidence = mesh._physical_evidence(
                    config, samples, report, truth, output
                )
                before_native = native_build_identity()
                mesh._study_figures(output, evidence)
                assert native_build_identity() == before_native
                assert mesh.hv._checksums(samples.output) == before_chains
                assert evidence["cost"]["retained_heat_adjoint_solves"] > 0
                assert (output / "conductivity.png").stat().st_size > 5000
                assert (output / "predictive-residuals.png").stat().st_size > 5000
        finally:
            if problem is not None:
                problem.close()
            base.close()
            _release_problem(base.forward.problem)


def test_mesh_extension_routes_larger_total_and_keeps_original_prefix(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext
    from dataclasses import replace

    study = _mesh_mechanics_only(monkeypatch)
    config = replace(study.hv.StudyConfig(), pilot_output=str(tmp_path / "pilot"))
    data = study.make_truth_data(config, truth="sharp", output=tmp_path / "truth")
    observations = study.hv._observations(config, data)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    study.hv._write(
        pilot / "manifest.json",
        {
            "identity": {
                "extra": {
                    "observation_sha256": study.hv._hash(
                        [a.tolist() for a in observations]
                    )
                }
            }
        },
    )
    np.savez(
        pilot / "observations.npz",
        points=observations[0],
        values=observations[1],
        sigma=observations[2],
    )
    monkeypatch.setattr(
        study.hv,
        "_qualified_pilot",
        lambda config: (
            {
                "meshes": {"16": {"tolerances": [1e-10, 1e-12], "spread": 0.1}},
                "initial_production_draws": 8,
            },
            "pilot-hash",
        ),
    )
    monkeypatch.setattr(study.hv, "_problem", lambda *args: nullcontext(None))
    monkeypatch.setattr(study.hv, "_optimize", lambda *args: (None, {"success": True}))
    targets = []
    report = SimpleNamespace(qualified=True)
    sample = SimpleNamespace(
        metadata={"total_draws": 8}, close=lambda: None, diagnostics=lambda: report
    )

    def sample_preserving_prefix(config, problem, initial, path, output, **kwargs):
        target = kwargs["draws"]
        targets.append(target)
        path.mkdir(exist_ok=True)
        artifact = path / "chains.sqlite"
        old = artifact.read_bytes() if artifact.exists() else b""
        following = b"".join(f"retained-{i}\n".encode() for i in range(target))
        assert following.startswith(old)
        artifact.write_bytes(following)
        sample.metadata["total_draws"] = target
        sample.output = path
        return sample, report

    monkeypatch.setattr(study.hv, "_converged_samples", sample_preserving_prefix)
    monkeypatch.setattr(
        study,
        "_physical_evidence",
        lambda *args: {"uncertainty": {"status": "unresolved"}},
    )
    monkeypatch.setattr(study, "_study_figures", lambda *args: None)
    monkeypatch.setattr(study.hv, "_chain_evidence", lambda *args: {"chains": []})
    output = tmp_path / "mesh"
    initial = study.run_mesh_study(config, data_path=data, nmesh=16, output=output)
    before = (output / "posterior/chains.sqlite").read_bytes()
    following = study.run_mesh_study(
        config, data_path=data, nmesh=16, output=output, total_draws=16
    )
    assert targets == [8, 16]
    assert following["generation"] == 1 and following["requested_total"] == 16
    assert (
        output / "history/generation-0000/posterior/chains.sqlite"
    ).read_bytes() == before
    assert (output / "posterior/chains.sqlite").read_bytes().startswith(before)
    assert (
        study._load_publication(output / "history/generation-0000/result.json")
        == initial
    )
    assert (
        study.run_mesh_study(config, data_path=data, nmesh=16, output=output)
        == following
    )
    import posterior_samples

    monkeypatch.setattr(posterior_samples.PosteriorSamples, "open", lambda path: sample)
    reanalyzed = study.run_mesh_study(
        config, data_path=data, nmesh=16, output=output, reanalyze=True
    )
    assert targets == [8, 16]  # No sampler call during reanalysis.
    assert reanalyzed["generation"] == 2 and reanalyzed["requested_total"] == 16


@pytest.mark.parametrize("complete_copy", [False, True])
def test_interrupted_archive_copy_can_retry_extension(
    tmp_path, monkeypatch, complete_copy
):
    artifact = tmp_path / "posterior/chains.sqlite"
    artifact.parent.mkdir()
    artifact.write_bytes(b"complete saved physical trajectory")
    previous = dict(
        status="sampled",
        generation=0,
        requested_total=8,
        checksums={"posterior/chains.sqlite": mesh.hv._file_hash(artifact)},
    )
    mesh._publish_record(tmp_path / "result.json", previous)
    copy = mesh.shutil.copy2

    def interrupt(source, destination):
        if complete_copy:
            copy(source, destination)
        else:
            Path(destination).write_bytes(b"partial trajectory")
        raise KeyboardInterrupt("archive copy interrupted")

    monkeypatch.setattr(mesh.shutil, "copy2", interrupt)
    with pytest.raises(KeyboardInterrupt, match="archive copy"):
        mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
    archived = tmp_path / "history/generation-0000/posterior/chains.sqlite"
    assert (
        not archived.exists()
    ), "An unverified archive must not acquire its final name."
    assert mesh._load_publication(tmp_path / "result.json") == previous
    assert not (tmp_path / "analysis-request.json").exists()
    monkeypatch.setattr(mesh.shutil, "copy2", copy)
    request, cached = mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
    assert request["generation"] == 1 and cached is None
    assert archived.read_bytes() == artifact.read_bytes()
    assert mesh._load_publication(archived.parent.parent / "result.json") == previous


@pytest.mark.parametrize("pending_name", ["result.json", "analysis-request.json"])
@pytest.mark.parametrize("historical", [False, True])
def test_report_cannot_recover_publication_owned_by_active_study(
    tmp_path, monkeypatch, pending_name, historical
):
    owner = tmp_path / "physical-study"
    output = owner / "history/generation-0000" if historical else owner
    output.mkdir(parents=True)
    identity = {
        "kind": "mesh",
        "extra": {
            "truth": "smooth",
            "nmesh": 16,
            "prior_case": "baseline",
            "truth_array_sha256": "shared-observations",
        },
    }
    digest = mesh.hv._hash(identity)
    mesh.hv._write(
        output / "manifest.json", {"identity": identity, "identity_sha256": digest}
    )
    previous = dict(
        status="failed",
        truth="smooth",
        nmesh=16,
        prior_case="baseline",
        manifest_sha256=digest,
        generation=0,
        checksums={},
    )
    mesh._publish_record(output / "result.json", previous)
    path = output / pending_name
    payload = (
        previous
        if pending_name == "result.json"
        else {"generation": 1, "total_draws": 16}
    )
    pending_path = mesh._publication_paths(path)[1]
    mesh.hv._write(
        pending_path,
        {
            "payload": payload,
            "sha256": mesh.hashlib.sha256(
                mesh.hv._encoded(payload) + b"\n"
            ).hexdigest(),
        },
    )
    before = {p.name: p.read_bytes() for p in output.glob("*.json")}
    destination = tmp_path / "comparison.md"
    with mesh.hv._study_lock(owner):
        with pytest.raises(RuntimeError, match="active writer"):
            mesh.write_study_report([output / "manifest.json"], destination)
        assert {p.name: p.read_bytes() for p in output.glob("*.json")} == before
    verify = mesh._verify_artifacts

    def verify_owned(directory, result):
        with pytest.raises(RuntimeError, match="active writer"):
            with mesh.hv._study_lock(owner):
                pass
        return verify(directory, result)

    monkeypatch.setattr(mesh, "_verify_artifacts", verify_owned)
    mesh.write_study_report(
        [output / "manifest.json", output / "manifest.json"], destination
    )
    assert not pending_path.exists()
    if historical:
        assert not (output / "study.lock").exists()
    assert (
        "failed" if pending_name == "result.json" else "extension_in_progress"
    ) in destination.read_text()


def test_failed_chain_fields_are_projected_before_reading_next_draw(
    tmp_path, monkeypatch
):
    import weakref

    @dataclass
    class Row:
        statistics: dict
        h: np.ndarray

    alive = []
    count = 100
    operator = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    monkeypatch.setattr(mesh.hv, "_quantity_operator", lambda record: operator)

    def records(chain):
        for i in range(count):
            assert (
                sum(ref() is not None for ref in alive) <= 1
            ), "Full conductivity draws accumulated in memory."
            row = Row(
                dict(
                    acceptance_probability=0.5 + 0.001 * i,
                    delta_energy=float(i),
                    trajectory_seconds=2.0,
                    solve_counts={"heat": 3, "adjoint": 2},
                ),
                np.array([1.0 + i, 2.0 + i, 3.0 + i]),
            )
            alive.append(weakref.ref(row))
            yield row
            del row

    def unavailable():
        raise ValueError("Uneven chains have no qualified diagnostics")

    samples = SimpleNamespace(
        metadata={"chains": 1},
        _record=object(),
        output=tmp_path / "posterior",
        _counts=lambda: np.array([count]),
        diagnostics=unavailable,
        _store=SimpleNamespace(
            records=records,
            failures=lambda: iter([]),
            checkpoint=lambda chain: SimpleNamespace(
                counters={"heat": 300, "adjoint": 200}
            ),
        ),
    )
    result = mesh._partial_chain_evidence(samples, tmp_path)
    assert result["acceptance"]["count"] == count
    assert result["energy_error"]["mean"] == 49.5
    assert result["cost"]["retained_seconds"] == 200.0
    assert result["cost"]["retained_heat_adjoint_solves"] == 500.0
    expected = np.array(
        [np.array([1.0 + i, 2.0 + i, 3.0 + i]) @ operator.T for i in range(count)]
    )[None, :, :]
    physical = result["autocorrelation"]["physical"][0]
    assert physical["lags"] == list(range(1, 65))
    np.testing.assert_allclose(
        physical["values"][0], mesh.hv._autocorrelation(expected, lag=1)[0]
    )


@pytest.mark.parametrize("counts", [(256, 256), (3, 2), (0, 0)])
def test_chain_report_releases_fields_between_draws_phases_and_chains(
    tmp_path, monkeypatch, counts
):
    """Scalar chain evidence must not retain the full conductivity histories."""
    from contextlib import nullcontext
    import weakref

    from posterior_store import RunStore, TrajectoryRecord

    alive = []

    def fields_released():
        assert not any(
            ref() is not None for ref in alive
        ), "Chain evidence retained full fields while reading the next draw."

    def records(chain, *, phase="sample"):
        count = counts[0 if phase == "warmup" else 1]
        for index in range(count):
            fields_released()
            row = TrajectoryRecord(
                chain=chain,
                phase=phase,
                index=index,
                inferred_coordinates={"z": np.zeros(3)},
                h=np.ones(3),
                sensor_temperature=np.full(2, 300.0),
                statistics={
                    "index": index,
                    "phase": phase,
                    "chain": chain,
                    "divergent": (
                        index == 191 + chain
                        if phase == "warmup"
                        else chain == 1 and index == 0
                    ),
                    "step_size": (
                        0.012 + chain * 0.001
                        if phase == "warmup" and index == 0
                        else 0.04 if phase == "warmup" else 0.02
                    ),
                    "acceptance_probability": 0.8 if chain == 0 else 0.6,
                },
            )
            alive.extend(
                weakref.ref(value)
                for value in (
                    row,
                    row.h,
                    row.inferred_coordinates["z"],
                    row.sensor_temperature,
                )
            )
            yield row
            del row
        fields_released()

    failures = [{"exception": "KeyboardInterrupt", "chain": 1}]
    streams = [{"initialization": [1, 2]}, {"initialization": [3, 4]}]
    store = SimpleNamespace(
        metadata={"chains": 2, "seed": 817, "random_streams": streams},
        problem_record=SimpleNamespace(
            arrays={
                "sensor_values": np.array([300.0, 302.0]),
                "sensor_sigma": np.array([2.0, 1.0]),
            }
        ),
        initial_checkpoint=lambda chain: SimpleNamespace(
            physical_state={"sensor_temperature": np.array([302.0 + 2 * chain, 304.0])}
        ),
        checkpoint=lambda chain: SimpleNamespace(
            step_size=0.03 + 0.01 * chain,
            counters={"heat": 100 + chain, "adjoint": 80 + chain},
        ),
        records=records,
        failures=lambda: iter(failures),
    )
    monkeypatch.setattr(RunStore, "open", lambda path: nullcontext(store))

    result = mesh.hv._chain_evidence(tmp_path / "posterior", lower_bound=0.01)

    fields_released()
    assert result["seed"] == 817
    assert result["random_streams"] == streams
    assert result["failures"] == failures
    assert result["passes_spread_criteria"] is False
    assert [item["initial_phi"] for item in result["chains"]] == [2.5, 4.0]
    assert [item["chain"] for item in result["chains"]] == [0, 1]
    for chain, item in enumerate(result["chains"]):
        assert item["warmup_count"] == counts[0]
        assert item["retained_count"] == counts[1]
        assert item["final_warmup_divergences"] == (
            1 if chain == 1 and counts[0] == 256 else 0
        )
        assert item["retained_divergences"] == (1 if chain == 1 and counts[1] else 0)
        assert item["mean_retained_acceptance"] == (
            pytest.approx(0.8 if chain == 0 else 0.6) if counts[1] else None
        )
        assert item["minimum_step_size"] == (
            0.012 + 0.001 * chain if counts[0] else 0.01
        )
        assert item["final_step_size"] == 0.03 + 0.01 * chain
        assert item["solve_counts"] == {"heat": 100 + chain, "adjoint": 80 + chain}
        assert item["passes_spread_criteria"] is (chain == 0 and counts == (256, 256))
        for phase, count in zip(("warmup", "retained"), counts):
            statistics = item["statistics"][phase]
            assert [s["index"] for s in statistics] == list(range(count))
            assert all(s["chain"] == chain for s in statistics)
            assert all(
                s["phase"] == ("warmup" if phase == "warmup" else "sample")
                for s in statistics
            )


def test_unverified_archive_copy_does_not_publish_and_retry_keeps_checked_evidence(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "posterior/chains.sqlite"
    artifact.parent.mkdir()
    artifact.write_bytes(b"complete saved physical trajectory")
    previous = dict(
        status="sampled",
        generation=0,
        requested_total=8,
        checksums={"posterior/chains.sqlite": mesh.hv._file_hash(artifact)},
    )
    mesh._publish_record(tmp_path / "result.json", previous)
    copy = mesh.shutil.copy2
    monkeypatch.setattr(
        mesh.shutil,
        "copy2",
        lambda source, destination: Path(destination).write_bytes(b"corrupt copy"),
    )
    with pytest.raises(ValueError, match="Staged.*checksum"):
        mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
    archived = tmp_path / "history/generation-0000/posterior/chains.sqlite"
    assert not archived.exists()
    monkeypatch.setattr(mesh.shutil, "copy2", copy)
    mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
    assert archived.read_bytes() == artifact.read_bytes()
    # A published generation remains immutable; it is not treated as a partial copy.
    archived.write_bytes(b"changed checked evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        mesh._prepare_analysis(tmp_path, total_draws=16, reanalyze=False)
