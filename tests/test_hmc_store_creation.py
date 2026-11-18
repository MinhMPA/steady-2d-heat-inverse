"""Creation must record one consistent numerical environment before publishing.

Only native provenance is supplied by a fixture. Source hashes, environment
comparison, physical records, SQLite transactions and reopening use real code.
"""

import copy
from dataclasses import replace
import json
import sqlite3

import pytest

import native_provenance
import posterior_store
from posterior_store import RunStore
from tests.test_hmc_store import metadata, problem_record


@pytest.fixture(autouse=True)
def controlled_native_provenance(monkeypatch):
    monkeypatch.setattr(
        native_provenance,
        "native_build_identity",
        lambda: {"libraries": [{"path": "controlled-libslepc", "sha256": "a" * 64}]},
    )


def declared_problem(environment):
    record = problem_record()
    return replace(
        record,
        metadata={**record.metadata, "numerical_environment": environment},
    )


def observe_once(monkeypatch, environment):
    observations = []

    def observe():
        observations.append(copy.deepcopy(environment))
        assert len(observations) == 1, "Creation must observe numerics only once."
        return copy.deepcopy(environment)

    monkeypatch.setattr(posterior_store, "numerical_environment", observe)
    return observations


def stored_creation(path):
    with sqlite3.connect(path / "chains.sqlite") as connection:
        row = connection.execute(
            "SELECT complete, initialized, environment FROM run WHERE id=1"
        ).fetchone()
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM trajectories").fetchone() == (
            0,
        )
    return row[0], row[1], posterior_store._unpack(row[2])


@pytest.mark.parametrize("difference", ["native_builds", "sources", "explicit_none"])
def test_creation_rejects_declared_environment_mismatch_preserves_evidence(
    tmp_path, monkeypatch, difference
):
    observed = posterior_store.numerical_environment()
    declared = copy.deepcopy(observed)
    if difference == "native_builds":
        # The historical saved record lacked SLEPc while its next observation had it.
        declared["native_builds"]["libraries"] = []
    elif difference == "sources":
        declared["sources"]["heat_model"] = "previous numerical source"
    else:
        declared = None
    problem = declared_problem(declared)
    observations = observe_once(monkeypatch, observed)
    path = tmp_path / "inconsistent"
    with pytest.raises(ValueError, match="numerical.*environment"):
        RunStore.create(path, problem=problem, run_metadata=metadata())
    assert observations == [observed]
    assert stored_creation(path) == (0, 0, observed)
    assert (
        json.loads((path / "problem.json").read_text())["metadata"][
            "numerical_environment"
        ]
        == declared
    )
    assert not (path / "status.json").exists()
    for writable in [False, True]:
        with pytest.raises(ValueError, match="incomplete"):
            RunStore.open(path, writable=writable)
    evidence = {
        name: (path / name).read_bytes()
        for name in ["problem.json", "problem.npz", "chains.sqlite"]
    }
    with pytest.raises(FileExistsError):
        RunStore.create(path, problem=problem, run_metadata=metadata())
    assert evidence == {name: (path / name).read_bytes() for name in evidence}
    assert observations == [observed]


@pytest.mark.parametrize("declare_environment", [False, True])
def test_creation_records_one_observation_and_supports_generic_records(
    tmp_path, monkeypatch, declare_environment
):
    observed = posterior_store.numerical_environment()
    problem = declared_problem(observed) if declare_environment else problem_record()
    observations = observe_once(monkeypatch, observed)
    path = tmp_path / "consistent"
    with RunStore.create(path, problem=problem, run_metadata=metadata()):
        assert stored_creation(path) == (1, 0, observed)
    with RunStore.open(path) as saved:
        assert saved.problem_record.metadata == problem.metadata
    assert observations == [observed]


def test_creation_observes_new_environment_on_each_call(tmp_path, monkeypatch):
    observed = posterior_store.numerical_environment()
    changed = copy.deepcopy(observed)
    changed["sources"]["heat_model"] = "next numerical source"
    environments = iter([observed, changed])
    monkeypatch.setattr(
        posterior_store, "numerical_environment", lambda: next(environments)
    )
    problem = declared_problem(observed)
    with RunStore.create(tmp_path / "first", problem=problem, run_metadata=metadata()):
        pass
    with pytest.raises(ValueError, match="numerical.*environment"):
        RunStore.create(tmp_path / "second", problem=problem, run_metadata=metadata())
    assert stored_creation(tmp_path / "first") == (1, 0, observed)
    assert stored_creation(tmp_path / "second") == (0, 0, changed)


@pytest.mark.parametrize("boundary", ["before_publication", "after_publication"])
def test_creation_interruption_preserves_publication_boundary(
    tmp_path, monkeypatch, boundary
):
    observed = posterior_store.numerical_environment()
    problem = declared_problem(observed)
    path = tmp_path / boundary

    def interrupt(*args):
        raise KeyboardInterrupt("controlled creation interruption")

    with monkeypatch.context() as changes:
        if boundary == "before_publication":
            changes.setattr(posterior_store, "_same", interrupt)
        else:
            changes.setattr(RunStore, "_write_status", interrupt)
        with pytest.raises(KeyboardInterrupt):
            RunStore.create(path, problem=problem, run_metadata=metadata())
    complete = int(boundary == "after_publication")
    assert stored_creation(path) == (complete, 0, observed)
    if complete:
        with RunStore.open(path, writable=True) as saved:
            assert saved.problem_record.metadata == problem.metadata
        assert (path / "status.json").is_file()
    else:
        with pytest.raises(ValueError, match="incomplete"):
            RunStore.open(path, writable=True)
    with pytest.raises(FileExistsError):
        RunStore.create(path, problem=problem, run_metadata=metadata())
