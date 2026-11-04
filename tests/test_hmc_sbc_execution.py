"""Indexed calibration ownership/restart mechanics; pure tests need no JAX."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from hmc_sbc_worker import issue_attempt, inspect_replica, coordinate, atomic_json


def experiment(path, replicas=3):
    path.mkdir()
    identity = {
        "kind": "sbc",
        "extra": {"replicates": replicas},
        "config": {"seed": 20261026},
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(
        path / "manifest.json", {"identity": identity, "identity_sha256": digest}
    )
    return digest


def valid_result(directory, result):
    assert result["fixture"] == "manufactured"
    assert result["scientific"]["index"] == result["replicate"]
    return result


def inspect(output, index, total, identity):
    return inspect_replica(
        output,
        index=index,
        replicates=total,
        identity=identity,
        validate_result=valid_result,
    )


def wait_for(path, timeout=15):
    until = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > until:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.03)
    return path


@pytest.fixture
def worker_script(tmp_path):
    source = r"""
import json, os, signal, sys, time
from pathlib import Path
from hmc_sbc_worker import worker_main, atomic_json

def work(output, index, total, identity, cancel):
    directory = output / f"replicate-{index:03d}"
    control_path = output / "fixture-control.json"
    controls = json.loads(control_path.read_text()) if control_path.exists() else {}
    mode = controls.get("mode", "normal")
    (directory / "work-count.txt").open("a").write("called\n")
    atomic_json(directory / "entered.json", {"pid": os.getpid()})
    if mode == "hold":
        while not (output / "release").exists():
            cancel("safe_fixture_boundary")
            time.sleep(0.02)
    if mode == "interrupt_once" and not (directory / "interrupted-once").exists():
        atomic_json(directory / "committed.json", {"index": index, "target": 16})
        (directory / "interrupted-once").touch()
        os.kill(os.getpid(), signal.SIGINT)
        cancel("committed_fixture_boundary")
    if mode == "resource_failure":
        import errno
        raise OSError(errno.ENOSPC, "manufactured unavailable storage")
    if mode == "crash":
        atomic_json(directory / "committed.json", {"index": index, "target": 16})
        os._exit(19)
    time.sleep(0.025*(total-index))
    # Deterministic indexed data, independent of process/scheduling identities.
    import numpy as np
    from hmc_validation import _seed
    from types import SimpleNamespace
    config = SimpleNamespace(seed=20261026)
    seeds = {label: _seed(config,30,index,k) for k,label in enumerate(("truth","noise","sampling","rank_ties"))}
    scientific = {"index": index, "truth": np.random.default_rng(seeds["truth"]).normal(size=4).tolist(),
        "noise": np.random.default_rng(seeds["noise"]).normal(size=4).tolist(),
        "draws": np.random.default_rng(seeds["sampling"]).normal(size=(4,8,4)).tolist(), "rank_seed": seeds["rank_ties"]}
    status = "failed" if mode == "scientific_failure" and index == 0 else "qualified"
    return {"status": status, "fixture": "manufactured", "seeds": seeds, "scientific": scientific}

# Exercise parent-loss window while a child has not yet acquired its lock.
if os.environ.get("SBC_TEST_PRELOCK"):
    gate = Path(os.environ["SBC_TEST_PRELOCK"])
    atomic_json(gate.with_suffix(".ready"), {"pid": os.getpid()})
    while not gate.exists(): time.sleep(0.02)
raise SystemExit(worker_main(work=work))
"""
    path = tmp_path / "fixture_worker.py"
    path.write_text(source)
    return [sys.executable, str(path)]


def run(output, total, workers, identity, command):
    return coordinate(
        output,
        replicates=total,
        workers=workers,
        identity=identity,
        validate_result=valid_result,
        command=command,
        poll_interval=0.02,
    )


def test_unclosed_issue_is_not_eligible_even_without_lock(tmp_path):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    issued = issue_attempt(
        output,
        index=0,
        replicates=1,
        identity=identity,
        session="first",
        command=[sys.executable],
    )
    for _ in range(2):
        state = inspect(output, 0, 1, identity)
        assert state["status"] == "unresolved" and state["reserved"]
        assert not state["eligible"] and state["attempt"] == str(issued)


def test_explicit_cooperative_closure_required(tmp_path):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    issued = issue_attempt(
        output,
        index=0,
        replicates=1,
        identity=identity,
        session="first",
        command=[sys.executable],
    )
    issue = json.loads(issued.read_text())
    directory = output / "replicate-000"
    directory.mkdir(exist_ok=True)
    closure = directory / f"attempt-{issue['attempt']}.closure.json"
    record = {
        "attempt": issue["attempt"],
        "replicate": 0,
        "replicates": 1,
        "identity": identity,
        "status": "interrupted",
        "cooperative": False,
        "stage": "before_science",
    }
    atomic_json(closure, record)
    assert not inspect(output, 0, 1, identity)["eligible"]
    record["cooperative"] = True
    atomic_json(closure, record)
    state = inspect(output, 0, 1, identity)
    assert state["eligible"] and not state["reserved"]
    record["attempt"] = "wrong"
    atomic_json(closure, record)
    assert inspect(output, 0, 1, identity)["status"] == "unresolved"


def test_serial_parallel_exact_indexed_science_and_reuse(tmp_path, worker_script):
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    results = []
    for path, workers in zip(outputs, (1, 2)):
        identity = experiment(path)
        result = run(path, 3, workers, identity, worker_script)
        assert [r["replicate"] for r in result] == [0, 1, 2]
        assert all(r["status"] == "qualified" for r in result)
        assert [r["scientific"]["index"] for r in result] == [0, 1, 2]
        assert len({seed for row in result for seed in row["seeds"].values()}) == 12
        assert run(path, 3, workers, identity, worker_script) == result
        assert all(
            (path / f"replicate-{i:03d}" / "work-count.txt").read_text() == "called\n"
            for i in range(3)
        )
        results.append(result)
    assert [r["scientific"] for r in results[0]] == [
        r["scientific"] for r in results[1]
    ]
    assert [r["seeds"] for r in results[0]] == [r["seeds"] for r in results[1]]


def test_cooperative_restart_preserves_existing_commit(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    atomic_json(output / "fixture-control.json", {"mode": "interrupt_once"})
    first = run(output, 1, 1, identity, worker_script)
    assert first[0]["status"] == "interrupted"
    committed = (output / "replicate-000/committed.json").read_bytes()
    second = run(output, 1, 1, identity, worker_script)
    assert second[0]["status"] == "qualified"
    assert (output / "replicate-000/committed.json").read_bytes() == committed
    assert len(list((output / "executions").glob("*/attempt-*.issued.json"))) == 2


def test_abnormal_exit_stays_unresolved_across_restart(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    atomic_json(output / "fixture-control.json", {"mode": "crash"})
    first = run(output, 1, 1, identity, worker_script)
    assert first[0]["status"] == "unresolved"
    committed = (output / "replicate-000/committed.json").read_bytes()
    atomic_json(output / "fixture-control.json", {"mode": "normal"})
    assert run(output, 1, 1, identity, worker_script)[0]["status"] == "unresolved"
    assert (output / "replicate-000/work-count.txt").read_text() == "called\n"
    assert (output / "replicate-000/committed.json").read_bytes() == committed


def test_scientific_failure_remains_and_other_indices_complete(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 3)
    atomic_json(output / "fixture-control.json", {"mode": "scientific_failure"})
    rows = run(output, 3, 2, identity, worker_script)
    assert [r["status"] for r in rows] == ["failed", "qualified", "qualified"]
    atomic_json(output / "fixture-control.json", {"mode": "normal"})
    assert run(output, 3, 1, identity, worker_script) == rows
    assert (output / "replicate-000/work-count.txt").read_text() == "called\n"


def test_duplicate_worker_never_enters_science(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    atomic_json(output / "fixture-control.json", {"mode": "hold"})
    issued = issue_attempt(
        output,
        index=0,
        replicates=1,
        identity=identity,
        session="manual",
        command=worker_script,
    )
    command = worker_script + [
        "--output",
        str(output),
        "--attempt",
        str(issued),
        "--index",
        "0",
        "--replicates",
        "1",
    ]
    winner = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        wait_for(output / "replicate-000/entered.json")
        loser = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert loser.returncode != 0
        (output / "release").touch()
        assert winner.wait(timeout=10) == 0
        assert (output / "replicate-000/work-count.txt").read_text() == "called\n"
    finally:
        if winner.poll() is None:
            winner.kill()
            winner.wait()


def test_issued_prelock_child_reserves_slot_after_parent_loss(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    issued = issue_attempt(
        output,
        index=0,
        replicates=1,
        identity=identity,
        session="lost-parent",
        command=worker_script,
    )
    gate = tmp_path / "prelock"
    command = worker_script + [
        "--output",
        str(output),
        "--attempt",
        str(issued),
        "--index",
        "0",
        "--replicates",
        "1",
    ]
    child = subprocess.Popen(
        command,
        env=dict(os.environ, SBC_TEST_PRELOCK=str(gate)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(gate.with_suffix(".ready"))
        assert inspect(output, 0, 1, identity)["reserved"]
        assert run(output, 1, 1, identity, worker_script)[0]["status"] == "unresolved"
        assert not (output / "replicate-000/work-count.txt").exists()
        gate.touch()
        assert child.wait(timeout=15) == 0
        assert run(output, 1, 1, identity, worker_script)[0]["status"] == "qualified"
        assert (output / "replicate-000/work-count.txt").read_text() == "called\n"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_terminal_result_without_closure_is_reused(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 1)
    rows = run(output, 1, 1, identity, worker_script)
    closures = list((output / "replicate-000").glob("attempt-*.closure.json"))
    assert len(closures) == 1
    closures[0].unlink()  # Emulate crash after atomic terminal publication.
    assert run(output, 1, 1, identity, worker_script) == rows
    assert (output / "replicate-000/work-count.txt").read_text() == "called\n"


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, 4])
def test_worker_limit_is_explicit_and_bounded(tmp_path, workers):
    output = tmp_path / "study"
    identity = experiment(output, 3)
    with pytest.raises(ValueError):
        run(output, 3, workers, identity, [sys.executable])


def test_parent_death_leaves_replica_owned_but_study_lock_available(
    tmp_path, worker_script
):
    import fcntl

    output = tmp_path / "study"
    identity = experiment(output, 1)
    atomic_json(output / "fixture-control.json", {"mode": "hold"})
    parent_code = """
import json,sys
from pathlib import Path
from hmc_validation import _study_lock
from hmc_sbc_worker import coordinate
output=Path(sys.argv[1])
with _study_lock(output):
    coordinate(output,replicates=1,workers=1,identity=sys.argv[2],
        validate_result=lambda directory,result:result,command=json.loads(sys.argv[3]),poll_interval=.02)
"""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(output),
            identity,
            json.dumps(worker_script),
        ],
        close_fds=True,
    )
    try:
        wait_for(output / "replicate-000/entered.json")
        parent.kill()
        parent.wait(timeout=10)
        with (output / "study.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert inspect(output, 0, 1, identity)["status"] == "active"
            fcntl.flock(stream, fcntl.LOCK_UN)
        (output / "release").touch()
        wait_for(output / "replicate-000/result.json")
        records = run(output, 1, 1, identity, worker_script)
        assert records[0]["status"] == "qualified"
        assert (output / "replicate-000/work-count.txt").read_text() == "called\n"
        assert len(list((output / "executions").glob("*/attempt-*.issued.json"))) == 1
    finally:
        (output / "release").touch()
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)


def test_known_resource_failure_stops_admission(tmp_path, worker_script):
    output = tmp_path / "study"
    identity = experiment(output, 3)
    atomic_json(output / "fixture-control.json", {"mode": "resource_failure"})
    records = run(output, 3, 1, identity, worker_script)
    assert records[0]["status"] == "unresolved"
    assert [row["status"] for row in records[1:]] == ["not_started", "not_started"]
    assert len(list((output / "executions").glob("*/attempt-*.issued.json"))) == 1
    assert run(output, 3, 1, identity, worker_script) == records
