"""Saved heat calculations preserve physical states and trajectory boundaries."""

# standard library imports
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

# numerical and testing imports
import numpy as np
import pytest

# local imports
from posterior_store import ChainCheckpoint, ProblemRecord, RunStore, TrajectoryRecord


def problem_record():
    return ProblemRecord(
        {"coordinate_schema": {"z": [3], "interface": [1]}},
        {"coordinates": np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])},
        "synthetic-physical-law",
        "synthetic-fixed-numerics",
    )


def checkpoint(phase="sample", done=0, chain=0):
    return ChainCheckpoint(
        chain=chain,
        phase=phase,
        warmup_done=done if phase == "warmup" else 0,
        draws_done=done if phase == "sample" else 0,
        inferred_coordinates={
            "z": np.array([0.2, -0.4, 0.1]),
            "interface": np.array([0.6]),
        },
        physical_state={
            "h": np.array([1.0, 2.0, 3.0]),
            "sensor_temperature": np.array([300.1, 300.2]),
        },
        hmc_state={
            "position": np.array([0.2, -0.4, 0.1]),
            "logdensity": np.array(-2.0, dtype=np.float32),
            "logdensity_grad": np.array([1.0, 2.0, 3.0]),
        },
        rng_state={
            "implementation": "threefry2x32",
            "key_data": np.array([42, 7 + done], dtype=np.uint32),
        },
        adaptation_state={
            "log_step_size": np.array(-3.0),
            "log_step_size_avg": -2.9,
            "step": 7 + done,
            "avg_error": 0.1,
            "mu": 0.4,
        },
        step_size=0.05,
        counters={
            "divergences": 3 + done,
            "heat_solves": 8 + done,
            "adjoint_solves": 7 + done,
            "accepted": 2,
        },
    )


def trajectory(cp, index):
    return TrajectoryRecord(
        cp.chain,
        cp.phase,
        index,
        cp.inferred_coordinates,
        cp.physical_state["h"],
        cp.physical_state["sensor_temperature"],
        {"accepted": False, "divergent": True, "energy_error": 1e4},
    )


def metadata():
    return {
        "total_draws": 4,
        "warmup": 0,
        "chains": 1,
        "seed": 73,
        "initialization": {"kind": "complete", "z": [0.2, -0.4, 0.1]},
        "controls": {"duration": [0.5, 1.5], "divergence_threshold": 1000.0},
    }


def assert_tree(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    else:
        assert type(left) is type(right)
        assert left == right


def test_complete_checkpoint_lossless_round_trip(tmp_path):
    output = tmp_path / "run"
    cp = checkpoint(done=0)
    with RunStore.create(
        output, problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([cp])
        following = checkpoint(done=1)
        store.commit_trajectory(trajectory(following, 0), following)
        for name in cp.__dataclass_fields__:
            assert_tree(getattr(following, name), getattr(store.checkpoint(0), name))
        for name in ("inferred_coordinates", "h", "sensor_temperature", "statistics"):
            assert_tree(
                getattr(trajectory(following, 0), name),
                getattr(list(store.records(0))[0], name),
            )
    with RunStore.open(output) as store:
        assert store.checkpoint(0).draws_done == 1
        assert_tree(problem_record().arrays, store.problem_record.arrays)
        with pytest.raises((PermissionError, RuntimeError)):
            store.update_target(8)


@pytest.mark.parametrize("phase", ["warmup", "sample"])
def test_atomic_trajectory_checkpoint_survive_process_death(tmp_path, phase):
    output = tmp_path / phase
    run_store_crash_child(output, phase)
    with RunStore.open(output) as store:
        rows = list(store.records(0, phase=phase))
        assert [row.index for row in rows] == [0]
        cp = store.checkpoint(0)
        assert (cp.warmup_done if phase == "warmup" else cp.draws_done) == 1
        assert_tree(rows[0].inferred_coordinates, cp.inferred_coordinates)
        assert_tree(cp.counters, checkpoint(phase, 1).counters)
    # The operating system releases the writer lock after abrupt process death.
    with RunStore.open(output, writable=True) as store:
        following = checkpoint(phase, 2)
        store.commit_trajectory(trajectory(following, 1), following)
        assert len(list(store.records(0, phase=phase))) == 2


def run_store_crash_child(path, phase):
    script = """
import os, sqlite3, sys
from tests.test_hmc_store import *
path, phase = sys.argv[1:]
with RunStore.create(path, problem=problem_record(), run_metadata={**metadata(), "warmup": 2 if phase == "warmup" else 0}) as s:
    s.initialize_chains([checkpoint(phase, 0)])
    cp = checkpoint(phase, 1)
    s.commit_trajectory(trajectory(cp, 0), cp)
s = RunStore.open(path, writable=True)
# Terminate after SQLite has executed the next trajectory insertion, with
# neither its checkpoint nor the transaction commit yet completed.
s._connection.create_function('die_now', 0, lambda: os._exit(83))
s._connection.execute('CREATE TEMP TRIGGER die_after_row AFTER INSERT ON trajectories BEGIN SELECT die_now(); END')
cp = checkpoint(phase, 2)
s.commit_trajectory(trajectory(cp, 1), cp)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), phase], capture_output=True, text=True
    )
    assert result.returncode == 83, result.stdout + result.stderr


def test_initialization_is_all_or_none_and_once(tmp_path):
    with RunStore.create(
        tmp_path / "run",
        problem=problem_record(),
        run_metadata={**metadata(), "chains": 2},
    ) as store:
        with pytest.raises(ValueError):
            store.initialize_chains([checkpoint(chain=0), checkpoint(chain=0)])
        with pytest.raises(KeyError):
            store.checkpoint(0)
        store.initialize_chains([checkpoint(chain=0), checkpoint(chain=1)])
        with pytest.raises((RuntimeError, ValueError)):
            store.initialize_chains([checkpoint(chain=0), checkpoint(chain=1)])
        with pytest.raises(ValueError):
            store.commit_trajectory(
                trajectory(checkpoint(done=1), 1), checkpoint(done=1)
            )
        wrong = replace(
            checkpoint(done=1),
            inferred_coordinates={"z": np.zeros(3), "interface": np.array([0.6])},
        )
        with pytest.raises(ValueError):
            store.commit_trajectory(trajectory(checkpoint(done=1), 0), wrong)
        assert list(store.records(0)) == []


def test_rejection_repeats_and_invalid_lengths(tmp_path):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        for index in range(3):
            cp = checkpoint(done=index + 1)
            store.commit_trajectory(trajectory(cp, index), cp)
        assert [r.index for r in store.records(0, start=1, stop=3)] == [1, 2]
        rows = list(store.records(0))
        assert all(np.array_equal(rows[0].h, row.h) for row in rows)
        cp = checkpoint(done=4)
        with pytest.raises(ValueError):
            store.commit_trajectory(replace(trajectory(cp, 3), h=np.ones(2)), cp)
        with pytest.raises(ValueError):
            store.commit_trajectory(
                replace(trajectory(cp, 3), inferred_coordinates={"z": np.ones(3)}), cp
            )
        assert store.checkpoint(0).draws_done == 3


def test_exclusive_writer_backup_and_status_recovery(tmp_path):
    path, backup = tmp_path / "run", tmp_path / "backup"
    with RunStore.create(
        path, problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        with pytest.raises((RuntimeError, BlockingIOError)):
            RunStore.open(path, writable=True)
        with RunStore.open(path) as reader:
            assert reader.checkpoint(0).draws_done == 0
        store.update_target(6)
        with pytest.raises(ValueError):
            store.update_target(5)
        store.write_failure({"stage": "heat", "reason": -3, "residual": np.array(1.2)})
        store.backup(backup)
        with pytest.raises(FileExistsError):
            store.backup(backup)
    (path / "status.json").write_text('{"draws_done": 999}')
    with RunStore.open(path, writable=True) as store:
        assert store.metadata["total_draws"] == 6
        assert (
            json.loads((path / "status.json").read_text())["chains"]["0"]["draws_done"]
            == 0
        )
    with RunStore.open(backup) as store:
        assert store.metadata["total_draws"] == 6
        assert store.checkpoint(0).draws_done == 0
        assert (
            sqlite3.connect(backup / "chains.sqlite")
            .execute("SELECT COUNT(*) FROM failures")
            .fetchone()[0]
            == 1
        )
    with pytest.raises(FileExistsError):
        RunStore.create(path, problem=problem_record(), run_metadata=metadata())


def test_invalid_serialization_and_problem_integrity(tmp_path):
    with pytest.raises((ValueError, TypeError)):
        ProblemRecord({}, {"bad": np.array([object()])}, "p", "n")
    with pytest.raises(ValueError):
        ProblemRecord({}, {"bad": np.array([np.nan])}, "p", "n")
    path = tmp_path / "run"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    document = json.loads((path / "problem.json").read_text())
    document["problem_id"] = "tampered"
    (path / "problem.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="hash|identity|integrity"):
        RunStore.open(path)
    document["schema_version"] = 999
    (path / "problem.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema|version"):
        RunStore.open(path)
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        RunStore.open(unfinished)


def test_changed_numerical_environment_rejects_append(tmp_path, monkeypatch):
    path = tmp_path / "run"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    monkeypatch.setenv("OMP_NUM_THREADS", "987")
    with RunStore.open(path):
        pass
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)


@pytest.mark.parametrize(
    "cell,nmesh",
    [("triangle", 3), ("triangle", 4), ("quadrilateral", 3), ("quadrilateral", 4)],
)
def test_frozen_heat_problem_reconstructs_in_fresh_process(tmp_path, cell, nmesh):
    from tests._hmc_helpers import make_problem, _release_problem
    from heat_inference import HeatInverseProblem

    path = tmp_path / "physical"
    original = make_problem(nmesh=nmesh, cell=cell)
    try:
        z = np.linspace(-0.2, 0.3, original.posterior.prior.dimension)
        expected = original.posterior.evaluate(z)
        with RunStore.create(
            path, problem=original.to_record(), run_metadata=metadata()
        ):
            pass
        np.save(tmp_path / "z.npy", z)
        with HeatInverseProblem.load(path) as loaded:
            assert loaded.forward is None
            assert loaded.problem_id == original.problem_id
            assert loaded.numerics_id == original.numerics_id
            assert np.array_equal(loaded._model.cells, original._model.cells)
            assert np.allclose(
                loaded.posterior.evaluate(z).sensor_temperature,
                expected.sensor_temperature,
                rtol=0,
                atol=2e-12,
            )
        script = """
import sys, numpy as np
from heat_inference import HeatInverseProblem
with HeatInverseProblem.load(sys.argv[1]) as p:
    assert p.forward is None
    ev = p.posterior.evaluate(np.load(sys.argv[2]))
    np.savez(sys.argv[3], h=ev.h, temperature=ev.sensor_temperature, gradient=ev.gradient)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                str(tmp_path / "z.npy"),
                str(tmp_path / "prediction.npz"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with np.load(tmp_path / "prediction.npz", allow_pickle=False) as values:
            assert np.array_equal(values["h"], expected.h)
            assert np.allclose(
                values["temperature"], expected.sensor_temperature, rtol=0, atol=2e-12
            )
            assert np.allclose(
                values["gradient"], expected.gradient, rtol=1e-9, atol=1e-10
            )
        record = original.to_record()
        for name, bad in [
            ("cells", np.array([[0.0, 1.0, 99999.0]])),
            ("coordinates", np.zeros((2, 3))),
        ]:
            arrays = {**record.arrays, name: bad}
            malformed = ProblemRecord(
                record.metadata, arrays, record.problem_id, record.numerical_id
            )
            with pytest.raises(ValueError):
                HeatInverseProblem.from_record(malformed)
    finally:
        original.close()
        _release_problem(original.forward.problem)


def test_phase_boundary_keeps_all_warmup_and_first_sample(tmp_path):
    with RunStore.create(
        tmp_path / "run",
        problem=problem_record(),
        run_metadata={**metadata(), "warmup": 2},
    ) as store:
        with pytest.raises(ValueError):
            store.initialize_chains([checkpoint("warmup", 1)])
        store.initialize_chains([checkpoint("warmup", 0)])
        for index in range(2):
            cp = checkpoint("warmup", index + 1)
            record = replace(trajectory(cp, index), statistics={"step_size": 0.1})
            # The next spacing is adapted after the completed trajectory.
            assert cp.step_size != record.statistics["step_size"]
            store.commit_trajectory(record, cp)
        cp = replace(
            checkpoint("sample", 1),
            warmup_done=2,
            counters=checkpoint("sample", 3).counters,
            rng_state=checkpoint("sample", 3).rng_state,
            adaptation_state=checkpoint("warmup", 2).adaptation_state,
        )
        store.commit_trajectory(trajectory(cp, 0), cp)
        assert [r.index for r in store.records(0, phase="warmup")] == [0, 1]
        assert [r.index for r in store.records(0)] == [0]
        assert store.checkpoint(0).phase == "sample"
        wrong = replace(checkpoint("warmup", 2), draws_done=1)
        with pytest.raises(ValueError):
            store.commit_trajectory(trajectory(wrong, 2), wrong)


def test_problem_and_full_source_identity_are_checked(tmp_path, monkeypatch):
    import posterior_store

    path = tmp_path / "run"
    with RunStore.create(
        path, problem=problem_record(), run_metadata=metadata()
    ) as store:
        with pytest.raises(ValueError):
            store.validate_identity(
                replace(problem_record(), problem_id="changed-physics")
            )
        with pytest.raises(ValueError):
            store.validate_identity(
                replace(problem_record(), numerical_id="changed-tolerance")
            )
        with pytest.raises(ValueError):
            store.validate_identity(
                replace(problem_record(), arrays={"coordinates": np.ones((3, 2))})
            )
        store.validate_identity(problem_record())
    current = posterior_store.numerical_environment()
    changed = {
        **current,
        "sources": {**current["sources"], "heat_model": "changed-numerical-source"},
    }
    monkeypatch.setattr(posterior_store, "numerical_environment", lambda: changed)
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("replacement", ["physical", "numerical", "contents"])
def test_database_is_bound_to_its_original_problem_files(
    tmp_path, writable, replacement
):
    import shutil

    first, second = tmp_path / "first", tmp_path / "second"
    original = problem_record()
    if replacement == "physical":
        changed = replace(original, problem_id="different-physical-experiment")
    elif replacement == "numerical":
        changed = replace(original, numerical_id="different-numerical-calculation")
    else:
        changed = replace(
            original, arrays={"coordinates": original.arrays["coordinates"] + 0.1}
        )
    for path, record in ((first, original), (second, changed)):
        with RunStore.create(path, problem=record, run_metadata=metadata()):
            pass
    # Both pairs are individually valid. They must never relabel the trajectory
    # database belonging to another physical/numerical calculation.
    for name in ("problem.json", "problem.npz"):
        shutil.copyfile(second / name, first / name)
    with pytest.raises(ValueError, match="identity|integrity|problem"):
        RunStore.open(first, writable=writable)


@pytest.mark.parametrize("chain", [0.0, False])
def test_trajectory_chain_label_must_be_an_integer(tmp_path, chain):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        cp = checkpoint(done=1)
        with pytest.raises(ValueError):
            store.commit_trajectory(replace(trajectory(cp, 0), chain=chain), cp)
        assert store.checkpoint(0).draws_done == 0
        assert list(store.records(0)) == []


def test_actual_initial_states_remain_after_completed_trajectories(tmp_path):
    path = tmp_path / "run"
    with RunStore.create(
        path, problem=problem_record(), run_metadata={**metadata(), "chains": 2}
    ) as store:
        starts = [checkpoint(chain=0), checkpoint(chain=1)]
        second_z = np.array([-0.3, 0.7, 0.1])
        starts[1] = replace(
            starts[1],
            inferred_coordinates={**starts[1].inferred_coordinates, "z": second_z},
            hmc_state={**starts[1].hmc_state, "position": second_z},
            rng_state={
                "implementation": "threefry2x32",
                "key_data": np.array([123, 456], dtype=np.uint32),
            },
        )
        store.initialize_chains(starts)
        for initial in starts:
            cp = replace(
                initial,
                draws_done=1,
                counters=checkpoint(done=1).counters,
                rng_state=checkpoint(done=1).rng_state,
            )
            store.commit_trajectory(trajectory(cp, 0), cp)
    with RunStore.open(path) as store:
        for initial in starts:
            saved = store.initial_checkpoint(initial.chain)
            for name in initial.__dataclass_fields__:
                assert_tree(getattr(initial, name), getattr(saved, name))
            assert store.checkpoint(initial.chain).draws_done == 1
            assert saved.draws_done == 0


def test_numpy_integer_counters_preserve_committed_boundaries(tmp_path):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        first = checkpoint()
        first = replace(first, counters={**first.counters, "divergences": np.int64(3)})
        store.initialize_chains([first])
        following = checkpoint(done=1)
        following = replace(
            following, counters={**following.counters, "divergences": np.int64(4)}
        )
        store.commit_trajectory(trajectory(following, 0), following)
        assert_tree(following.counters, store.checkpoint(0).counters)
        assert store.checkpoint(0).draws_done == 1


@pytest.mark.parametrize("mapping_name", ["hmc_state", "rng_state", "adaptation_state"])
def test_complete_continuation_mapping_cannot_drop_a_named_field(
    tmp_path, mapping_name
):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        following = checkpoint(done=1)
        mapping = dict(getattr(following, mapping_name))
        mapping.pop(next(iter(mapping)))
        following = replace(following, **{mapping_name: mapping})
        with pytest.raises(ValueError, match="field|structure|mapping|continuation"):
            store.commit_trajectory(trajectory(following, 0), following)
        assert store.checkpoint(0).draws_done == 0
        assert list(store.records(0)) == []


def test_continuation_scalar_can_become_a_zero_dimensional_array(tmp_path):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        following = checkpoint(done=1)
        following = replace(
            following,
            adaptation_state={
                **following.adaptation_state,
                "avg_error": np.array(0.1, dtype=np.float64),
            },
        )
        store.commit_trajectory(trajectory(following, 0), following)
        assert_tree(following.adaptation_state, store.checkpoint(0).adaptation_state)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_evaluated_spatial_inputs_and_reference_units_reconstruct(tmp_path, cell):
    from tests._hmc_helpers import make_forward, _release_problem
    from conductivity_prior import GaussianLogConductivity
    from heat_inference import HeatInverseProblem
    from temperature_observations import TemperatureObservations

    forward = make_forward(
        nmesh=4, cell=cell, q=lambda x: 1 + 0.3 * x[0] + 0.2 * x[1], bottom=299.0
    )
    observations = TemperatureObservations(
        np.array([[0.21, 0.33], [0.64, 0.72]]),
        np.array([299.1, 299.2]),
        np.array([0.02, 0.03]),
    )
    try:
        with HeatInverseProblem(
            forward=forward,
            observations=observations,
            prior=GaussianLogConductivity(
                mu=lambda x: -0.2 + 0.4 * x[0] - 0.1 * x[1],
                kappa=1.7,
                tau=0.8,
                h_ref=2.5,
            ),
        ) as original:
            record = original.to_record()
            z = np.linspace(-0.3, 0.5, original.posterior.prior.dimension)
            expected = original.posterior.evaluate(z)
            # Callables are gone: changing caller source cannot affect either
            # the frozen record or its reconstructed physical posterior.
            forward.q.function.x.array[:] = 9.0
            forward.q.function.x.scatter_forward()
            with HeatInverseProblem.from_record(record) as loaded:
                actual = loaded.posterior.evaluate(z)
                assert loaded.forward is None
                assert loaded.problem_id == original.problem_id
                assert np.array_equal(actual.h, expected.h)
                assert np.allclose(
                    actual.sensor_temperature,
                    expected.sensor_temperature,
                    rtol=0,
                    atol=2e-12,
                )
                assert np.allclose(
                    actual.gradient, expected.gradient, rtol=1e-9, atol=1e-10
                )
                prior = loaded.posterior.prior
                assert np.array_equal(actual.h, np.exp(np.log(2.5) + prior.to_log_h(z)))
    finally:
        _release_problem(forward.problem)


@pytest.mark.parametrize(
    "chain,start,stop", [(np.int64(0), 0, 3), (0, np.int64(1), 3), (0, 0, np.int64(2))]
)
def test_numpy_integer_selection_matches_python_draw_indices(
    tmp_path, chain, start, stop
):
    with RunStore.create(
        tmp_path / "run", problem=problem_record(), run_metadata=metadata()
    ) as store:
        store.initialize_chains([checkpoint()])
        for index in range(3):
            cp = checkpoint(done=index + 1)
            store.commit_trajectory(trajectory(cp, index), cp)
        expected = list(store.records(int(chain), start=int(start), stop=int(stop)))
        actual = list(store.records(chain, start=start, stop=stop))
        assert [row.index for row in actual] == [row.index for row in expected]
        for actual_row, expected_row in zip(actual, expected):
            assert_tree(
                actual_row.inferred_coordinates, expected_row.inferred_coordinates
            )
            assert_tree(actual_row.h, expected_row.h)


def test_changed_form_compiler_version_rejects_continuation(tmp_path, monkeypatch):
    import posterior_store

    actual_version = posterior_store.importlib.metadata.version
    actual_ffcx = actual_version("fenics-ffcx")
    original_environment = posterior_store.numerical_environment()
    path = tmp_path / "run"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass

    def changed_version(name):
        if name == "fenics-ffcx":
            return actual_ffcx + "+changed-form-compiler"
        return actual_version(name)

    # FFCx constructs the discrete prior and heat/adjoint forms. Changing only
    # that dependency must prevent continuation, while inspection stays portable.
    monkeypatch.setattr(posterior_store.importlib.metadata, "version", changed_version)
    with RunStore.open(path) as reader:
        assert reader.problem_record.problem_id == problem_record().problem_id
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)
    assert original_environment["versions"]["fenics-ffcx"] == actual_ffcx
