"""Native numerical identities must be inspectable and prevent changed-build append."""

import copy
import pytest
import native_provenance
from posterior_store import RunStore, numerical_environment
from tests.test_hmc_store import problem_record, metadata


def test_native_identity_reports_actual_compiler_and_library_bytes():
    evidence = native_provenance.native_build_identity()
    assert evidence["compiler"]["CC"]["version"]
    assert evidence["compiler"]["CC"]["executable_sha256"]
    assert evidence["libraries"]
    assert all(len(lib["sha256"]) == 64 for lib in evidence["libraries"])
    assert all(lib["path"] for lib in evidence["libraries"])
    builds = evidence["conda_builds"]
    # Conda builds are concrete evidence when present; other installations may
    # provide the same loaded components through system or wheel packages.
    for name, build in builds.items():
        assert build["build"]
        assert len(build["record_sha256"]) == 64
        if build["sha256"] is not None:
            assert len(build["sha256"]) == 64
    if any(name in builds for name in ("mpich", "openmpi")):
        assert any(
            builds[name]["build"] for name in ("mpich", "openmpi") if name in builds
        )
    assert (
        evidence["resolution"]
        == "fresh_process_verified_against_loaded_parent_components"
    )
    assert all(item["status"] == "loaded" for item in evidence["imports"].values())
    assert numerical_environment()["native_builds"] == evidence


@pytest.mark.parametrize("component", ["compiler", "libraries", "conda_builds"])
def test_changed_native_build_rejects_append_preserves_inspection(
    tmp_path, monkeypatch, component
):
    path = tmp_path / "native-change"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    actual = native_provenance.native_build_identity()
    changed = copy.deepcopy(actual)
    changed[component] = {"changed": "deliberate changed-build evidence"}
    monkeypatch.setattr(native_provenance, "native_build_identity", lambda: changed)
    with RunStore.open(path) as reader:
        assert reader.problem_record.problem_id == problem_record().problem_id
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)


def test_same_resolved_native_path_changed_bytes_reject_append(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    library = tmp_path / "libpetsc-controlled.dylib"
    library.write_bytes(b"first numerical build")
    compiler = native_provenance._compiler("CC")
    monkeypatch.setattr(native_provenance, "_compiler", lambda name: compiler)
    monkeypatch.setattr(native_provenance, "_loaded_images", lambda: [])

    def resolved_child(*args, **kwargs):
        evidence = {"imports": {}, "images": [str(library)]}
        return SimpleNamespace(
            returncode=0, stderr="", stdout="S2DHI_NATIVE_JSON=" + json.dumps(evidence)
        )

    monkeypatch.setattr(native_provenance.subprocess, "run", resolved_child)
    path = tmp_path / "run"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    library.write_bytes(b"different numerical build at the same resolved path")
    with RunStore.open(path) as reader:
        assert reader.problem_record.problem_id == problem_record().problem_id
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)


def test_changed_installed_sampling_binary_rejects_append_without_import(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    artifact = tmp_path / "controlled-jax-kernel.so"
    artifact.write_bytes(b"first installed kernel")
    actual_distribution = native_provenance.importlib.metadata.distribution

    def distribution(name):
        if name == "jaxlib":
            return SimpleNamespace(
                files=["controlled-jax-kernel.so"],
                version="test-build",
                locate_file=lambda value: artifact,
            )
        return actual_distribution(name)

    monkeypatch.setattr(
        native_provenance.importlib.metadata, "distribution", distribution
    )
    before = {
        name
        for name in sys.modules
        if name.split(".")[0] in ("jax", "jaxlib", "blackjax")
    }
    evidence = native_provenance.native_build_identity()
    assert (
        evidence["optional_sampling_artifacts"]["jaxlib"]["status"]
        == "installed_not_loaded"
    )
    path = tmp_path / "sampling-build-change"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    artifact.write_bytes(b"other installed kernel, unchanged distribution version")
    with RunStore.open(path) as reader:
        assert reader.problem_record.problem_id == problem_record().problem_id
    with pytest.raises(ValueError, match="numerical|environment"):
        RunStore.open(path, writable=True)
    after = {
        name
        for name in sys.modules
        if name.split(".")[0] in ("jax", "jaxlib", "blackjax")
    }
    assert after == before


@pytest.mark.parametrize("search_input", ["PATH", "PYTHONPATH", "CONDA_PREFIX"])
def test_unused_search_location_does_not_change_numerics_or_block_continuation(
    tmp_path, monkeypatch, search_input
):
    """Session/search-directory changes preserve the same resolved calculation."""
    import os

    unused = tmp_path / "unused-session-directory"
    unused.mkdir()
    original = native_provenance.native_build_identity()
    path = tmp_path / "unchanged-calculation"
    with RunStore.create(path, problem=problem_record(), run_metadata=metadata()):
        pass
    if search_input == "CONDA_PREFIX":
        # The active interpreter's sys.prefix, not a shell label, owns the stack.
        monkeypatch.setenv(search_input, str(unused))
    else:
        previous = os.environ.get(search_input)
        monkeypatch.setenv(
            search_input,
            str(unused) + (os.pathsep + previous if previous else ""),
        )
        if search_input == "PYTHONPATH":
            # Exercise actual fresh-process source search as well as the env label.
            monkeypatch.syspath_prepend(str(unused))
    current = native_provenance.native_build_identity()
    for component in (
        "compiler",
        "imports",
        "libraries",
        "conda_builds",
        "optional_sampling_artifacts",
        "interpreter",
        "prefix",
    ):
        assert current[component] == original[component]
    with RunStore.open(path, writable=True) as continuation:
        assert continuation.problem_record.problem_id == problem_record().problem_id
    assert current == original
