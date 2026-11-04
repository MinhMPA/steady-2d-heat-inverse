"""Small heat calculation in fresh replica processes; no calibration claim."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


def test_two_fresh_heat_replicas_match_serial_and_parallel(tmp_path):
    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip("optional full-field sampling qualification")
    from hmc_blackjax import require_sampling_environment

    require_sampling_environment()
    from hmc_sbc_worker import atomic_json, coordinate
    from hmc_validation import _hash, _validate_resume_artifacts

    worker = tmp_path / "toy_heat_worker.py"
    worker.write_text(r"""
import os,signal
import numpy as np
from hmc_sbc_worker import worker_main

def work(output,index,total,identity,cancel):
    # Deliberately toy-only controls: four chains, 4 warmup, 6 retained draws.
    from hmc_validation import StudyConfig,_observations,_problem,_sample,_chain_evidence,_sbc_seeds,_save_arrays,_json_value
    config=StudyConfig(sigma=.05,initial_step_size=.01,step_size_bounds=(1e-6,.02))
    seeds=_sbc_seeds(config,index)
    directory=output/f"replicate-{index:03d}"
    observations=_observations(config)
    generated=directory/"generated.npz"
    if not generated.exists():
        with _problem(config,2,observations,(1e-10,1e-12)) as generator:
            z=np.random.default_rng(seeds["truth"]).normal(size=generator.posterior.prior.dimension)
            h=generator.posterior.prior.to_h(z)
            _, prediction=generator.posterior.predict_h(h)
        data=(observations[0],prediction+np.random.default_rng(seeds["noise"]).normal(0,.05,len(prediction)),observations[2])
        _save_arrays(directory/"generated.npz",directory/"generation.json",{"truth":h,"values":data[1]}, {"seeds":seeds})
    with np.load(generated) as saved:
        h=saved["truth"].copy()
        data=(observations[0],saved["values"].copy(),observations[2])
    generation_before=generated.read_bytes()
    original_cancel=cancel
    def cancel(stage):
        marker=directory/"interrupted-once"
        if (output/"interrupt-first").exists() and index==0 and stage.startswith("committed:") and not marker.exists():
            marker.touch()
            os.kill(os.getpid(),signal.SIGINT)
        original_cancel(stage)
    with _problem(config,2,data,(1e-10,1e-12)) as problem:
        initial=problem.initial_state(2.)
        with _sample(config,problem,initial,directory/"posterior",directory,seed=seeds["sampling"],spread=.1,warmup=4,draws=6,cancel_check=cancel) as samples:
            draws=samples.load_draws()
            qualified=samples.diagnostics().qualified
        with _sample(config,problem,initial,directory/"posterior",directory,seed=seeds["sampling"],spread=.1,warmup=4,draws=6,cancel_check=cancel) as samples:
            assert np.array_equal(samples.load_draws(),draws)
    assert generated.read_bytes()==generation_before
    evidence=_chain_evidence(directory/"posterior",config.step_size_bounds[0])
    np.savez(directory/"comparison.npz",truth=h,data=data[1],draws=draws)
    # A manufactured terminal envelope exercises process collection only.
    return _json_value({"status":"qualified","fixture":"toy_heat_only","scientifically_qualified":qualified,"pid":os.getpid(),"seeds":seeds,"evidence":evidence})
raise SystemExit(worker_main(work=work))
""")
    results = []
    paths = []
    for label, workers in (("serial", 1), ("parallel", 2)):
        output = tmp_path / label
        output.mkdir()
        identity = {
            "kind": "sbc",
            "extra": {"replicates": 2},
            "fixture": "n2_sigma05_4warmup_6draws",
        }
        digest = _hash(identity)
        atomic_json(
            output / "manifest.json", {"identity": identity, "identity_sha256": digest}
        )

        def validate(directory, result):
            assert result["fixture"] == "toy_heat_only"
            assert not result["scientifically_qualified"]
            assert (directory / "comparison.npz").exists()
            return result

        if label == "serial":
            (output / "interrupt-first").touch()
            first = coordinate(
                output,
                replicates=2,
                workers=workers,
                identity=digest,
                validate_result=validate,
                validate_resume=_validate_resume_artifacts,
                command=[sys.executable, str(worker)],
                poll_interval=0.02,
            )
            assert first[0]["status"] == "interrupted"
            assert first[1]["status"] == "qualified"
            saved_generation = (output / "replicate-000/generated.npz").read_bytes()
        records = coordinate(
            output,
            replicates=2,
            workers=workers,
            identity=digest,
            validate_result=validate,
            validate_resume=_validate_resume_artifacts,
            command=[sys.executable, str(worker)],
            poll_interval=0.02,
        )
        assert all(record["status"] == "qualified" for record in records)
        if label == "serial":
            assert (
                output / "replicate-000/generated.npz"
            ).read_bytes() == saved_generation
        assert len({record["pid"] for record in records}) == 2
        assert all(record["pid"] != os.getpid() for record in records)
        assert (
            coordinate(
                output,
                replicates=2,
                workers=workers,
                identity=digest,
                validate_result=validate,
                command=[sys.executable, str(worker)],
            )
            == records
        )
        for record in records:
            assert len(record["evidence"]["random_streams"]) == 4
            for chain in record["evidence"]["chains"]:
                assert chain["warmup_count"] == 4 and chain["retained_count"] == 6
                assert np.isfinite(chain["initial_phi"])
                assert (
                    chain["solve_counts"]["heat"] > 0
                    and chain["solve_counts"]["adjoint"] > 0
                )
                for row in chain["statistics"]["retained"]:
                    assert np.pi / 4 <= row["duration"] <= 3 * np.pi / 4
        results.append(records)
        paths.append(output)
    assert [r["seeds"] for r in results[0]] == [r["seeds"] for r in results[1]]
    for index in range(2):
        with (
            np.load(paths[0] / f"replicate-{index:03d}/comparison.npz") as serial,
            np.load(paths[1] / f"replicate-{index:03d}/comparison.npz") as parallel,
        ):
            for name in ("truth", "data", "draws"):
                assert np.array_equal(serial[name], parallel[name])
