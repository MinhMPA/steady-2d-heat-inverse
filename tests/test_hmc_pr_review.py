"""Regressions for numerical identity, exhausted resources and scientific plots."""

import ast
from contextlib import nullcontext
import errno
import gc
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from ._hmc_helpers import make_problem, _release_problem
from .test_hmc_optimization import release

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def problem():
    value = make_problem(nmesh=2)
    try:
        yield value
    finally:
        release(value)


@pytest.mark.parametrize("resolver", ["jit", "ffcx"])
def test_compiler_changes_reject_append_without_relabelling_old_forms(
    problem, tmp_path, monkeypatch, resolver
):
    from dolfinx import jit
    from ffcx import options
    from posterior_store import RunStore

    record = problem.to_record()
    output = tmp_path / "run"
    with RunStore.create(output, problem=record, run_metadata={"chains": 1, "total_draws": 1}):
        pass
    target = jit if resolver == "jit" else options
    original = target.get_options

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if resolver == "jit":
            result["cffi_extra_compile_args"] = ["-O0"]
        else:
            result["table_rtol"] *= 2
        return result

    monkeypatch.setattr(target, "get_options", changed)
    with pytest.raises(ValueError, match="compiler settings changed"):
        problem.to_record()
    with pytest.raises(ValueError, match="compiler settings changed"):
        problem._new_posterior(prefix="changed_compiler_")
    with pytest.raises(ValueError, match="environment"):
        RunStore.open(output, writable=True)
    with RunStore.open(output) as archive:
        assert archive.metadata["total_draws"] == 1


def test_fresh_processes_record_effective_local_and_user_compiler_options(tmp_path, monkeypatch):
    config = tmp_path / "config"
    (config / "dolfinx").mkdir(parents=True)
    (config / "ffcx").mkdir()
    (config / "dolfinx/dolfinx_jit_options.json").write_text(json.dumps({"cffi_extra_compile_args": ["-O1"]}))
    (config / "ffcx/ffcx_options.json").write_text(json.dumps({"epsilon": 1e-12}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    records = []
    for index in range(2):
        working = tmp_path / str(index)
        working.mkdir()
        if index:
            (working / "dolfinx_jit_options.json").write_text(json.dumps({"cffi_extra_compile_args": ["-O2", "-ffast-math"]}))
            (working / "ffcx_options.json").write_text(json.dumps({"epsilon": 1e-6}))
        run = subprocess.run(
            [sys.executable, "-c", "import json; from posterior_store import form_compiler_options; print(json.dumps(form_compiler_options()))"],
            cwd=working, capture_output=True, text=True, timeout=60, check=True,
        )
        records.append(json.loads(run.stdout))
    assert records[0]["dolfinx_jit"]["cffi_extra_compile_args"] == ["-O1"]
    assert records[0]["ffcx"]["epsilon"] == 1e-12
    assert records[1]["dolfinx_jit"]["cffi_extra_compile_args"] == ["-O2", "-ffast-math"]
    assert records[1]["ffcx"]["epsilon"] == 1e-6


@pytest.mark.parametrize("phase", ["initial", "continue"])
@pytest.mark.parametrize("equation", ["heat", "adjoint"])
def test_private_solver_options_cannot_change_between_chains(problem, phase, equation):
    from petsc4py import PETSc

    prefix = f"hmc_chain_0_{phase}_"
    key = prefix + equation + "_ksp_norm_type"
    options = PETSc.Options()
    options[key] = "unpreconditioned"
    try:
        with pytest.raises(ValueError, match="HeatNumerics"):
            problem._new_posterior(prefix=prefix)
        # The caller's global configuration is not consumed or erased.
        assert options[key] == "unpreconditioned"
    finally:
        del options[key]


def test_closed_chain_releases_fields_even_when_handle_is_retained(problem):
    posterior = problem._new_posterior(prefix="review_release_")
    field = weakref.ref(posterior._adjoint)
    form = weakref.ref(posterior._gradient_form)
    handle = weakref.ref(posterior)
    posterior.close()
    gc.collect()
    assert field() is None and form() is None
    del posterior
    gc.collect()
    assert handle() is None
    assert len(problem._workspaces) == 1
    z = np.zeros(problem.posterior.prior.dimension)
    assert np.isfinite(problem.posterior.evaluate(z).potential)


def test_repeated_posterior_construction_leaves_global_options_unchanged():
    from petsc4py import PETSc

    before = PETSc.Options().getAll()
    for _ in range(2):
        value = make_problem(nmesh=2)
        try:
            assert PETSc.Options().getAll() == before
        finally:
            release(value)
    assert PETSc.Options().getAll() == before


@pytest.mark.parametrize("invalid", [False, True])
def test_temporary_solver_options_restore_existing_values_after_setup(invalid):
    from petsc4py import PETSc
    from forward_solver import _linear_problem

    value = make_problem(nmesh=2)
    options = PETSc.Options()
    prefix = "review_existing_options_"
    key = prefix + "ksp_type"
    options[key] = "cg"
    before = options.getAll()
    built = None
    try:
        forward = value.forward
        settings = {"ksp_type": "not_a_valid_solver" if invalid else "cg", "ksp_rtol": 1e-9}
        if invalid:
            with pytest.raises(PETSc.Error):
                _linear_problem(forward.a, forward.L, bcs=forward.bcs, petsc_options_prefix=prefix, petsc_options=settings)
        else:
            built = _linear_problem(forward.a, forward.L, bcs=forward.bcs, petsc_options_prefix=prefix, petsc_options=settings)
        assert options.getAll() == before
        # A following construction also proves no global prefix was left pushed.
        follow = make_problem(nmesh=2)
        release(follow)
        assert options.getAll() == before
    finally:
        if built is not None:
            _release_problem(built)
        del options[key]
        release(value)


def test_legacy_forward_keeps_requested_initial_temperature():
    from forward_solver import SteadyHeat2DForwardSolver

    forward = SteadyHeat2DForwardSolver(
        nmesh=3, h=1., q=1., DBC_value=300.,
        petsc_opts={"ksp_initial_guess_nonzero": True},
    )
    try:
        assert forward.problem.solver.getInitialGuessNonzero()
        forward.solve()
        first = forward.T.x.array.copy()
        forward.solve()
        assert forward.problem.solver.getInitialGuessNonzero()
        np.testing.assert_allclose(forward.T.x.array, first, atol=1e-8, rtol=0)
        assert forward.total_linear_solves == 2
    finally:
        _release_problem(forward.problem)


def exhausted(kind):
    if kind == "memory":
        return MemoryError("manufactured memory exhaustion")
    if kind in ("ENOMEM", "ENOSPC", "EDQUOT"):
        return OSError(getattr(errno, kind), "manufactured resource exhaustion")
    exc = sqlite3.OperationalError("manufactured SQLite exhaustion")
    exc.sqlite_errorcode = getattr(sqlite3, kind)
    return exc


@pytest.mark.parametrize("kind", ["memory", "ENOMEM", "ENOSPC", "EDQUOT", "SQLITE_FULL", "SQLITE_NOMEM", "SQLITE_IOERR_NOMEM"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_resource_classification_preserves_chained_native_failures(kind, wrapped):
    from hmc_sbc_worker import resource_exhausted

    exc = exhausted(kind)
    if wrapped:
        outer = RuntimeError("native calculation could not finish")
        outer.__cause__ = exc
        exc = outer
    assert resource_exhausted(exc)
    assert not resource_exhausted(ValueError("ordinary numerical failure"))


@pytest.mark.parametrize("kind", ["memory", "ENOMEM", "ENOSPC", "EDQUOT", "SQLITE_FULL", "SQLITE_NOMEM"])
def test_pilot_stops_before_admitting_another_spread(tmp_path, monkeypatch, kind):
    import hmc_validation as hv

    output = tmp_path / "pilot"
    output.mkdir()
    monkeypatch.setattr(hv, "_observations", lambda *a: (np.zeros((1, 2)), np.zeros(1), np.ones(1)))
    monkeypatch.setattr(hv, "_study_manifest", lambda *a, **k: {})
    monkeypatch.setattr(hv, "_require_native", lambda *a: None)
    monkeypatch.setattr(hv, "_problem", lambda *a, **k: nullcontext(SimpleNamespace(problem_id="fixture")))
    monkeypatch.setattr(hv, "_optimize", lambda *a: (SimpleNamespace(coordinates={"z": np.zeros(2)}), {"success": True}))
    monkeypatch.setattr(hv, "_force_comparison", lambda *a: {"passing_tolerance_indices": [0]})
    admitted = []

    def sample(*args, **kwargs):
        admitted.append(kwargs["spread"])
        raise exhausted(kind)

    monkeypatch.setattr(hv, "_sample", sample)
    with pytest.raises((MemoryError, OSError, sqlite3.OperationalError)):
        hv.run_numerical_pilot(hv.StudyConfig(), output)
    assert admitted == [hv.SPREADS[0]]
    assert (output / "failure.json").exists()
    assert not list(output.glob("mesh-*/spread-*-evidence.json"))
    with pytest.raises(RuntimeError, match="Recorded pilot failure"):
        hv.run_numerical_pilot(hv.StudyConfig(), output)
    assert len(admitted) == 1


@pytest.mark.parametrize("working_directory", ["", "notebooks"])
def test_notebook_reports_pending_generation_from_either_directory(tmp_path, monkeypatch, capsys, working_directory):
    import hmc_mesh_study as mesh

    (tmp_path / "src").mkdir()
    (tmp_path / "notebooks").mkdir()
    output = tmp_path / "runs" / "manufactured-extension"
    output.mkdir(parents=True)
    identity = {"kind": "mesh", "extra": {"truth": "smooth", "nmesh": 16, "prior_case": "baseline", "truth_array_sha256": "same-data"}}
    digest = mesh.hv._hash(identity)
    mesh.hv._write(output / "manifest.json", {"identity": identity, "identity_sha256": digest})
    mesh._publish_record(output / "result.json", {"status": "sampled", "truth": "smooth", "nmesh": 16, "prior_case": "baseline", "manifest_sha256": digest, "generation": 0, "coverage": {"sentinel": "obsolete coverage"}, "shared_locations": [], "uncertainty": {"status": "resolved"}, "checksums": {}})
    mesh._publish_record(output / "analysis-request.json", {"generation": 1, "total_draws": 16})
    notebook = json.loads((ROOT / "notebooks/HMCSampling.ipynb").read_text())
    monkeypatch.chdir(tmp_path / working_directory)
    namespace = {"Path": Path}
    # Execute the notebook's actual project-root expression and reporting cell.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "root" for t in node.targets):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-root", "exec"), namespace)
    exec("".join(notebook["cells"][-1]["source"]), namespace)
    shown = capsys.readouterr().out
    assert "extension_in_progress generation 1" in shown
    assert "obsolete coverage" not in shown
    assert (tmp_path / "runs/hmc-mesh-report.md").exists()


@pytest.mark.parametrize("n", [15, 16])
@pytest.mark.parametrize("cross", [False, True])
def test_notebook_radial_spectrum_matches_complete_fourier_plane(n, cross):
    notebook = json.loads((ROOT / "notebooks/MultiFidelitySolve.ipynb").read_text())
    namespace = {"np": np}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name == "power_spectrum":
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-spectrum", "exec"), namespace)
    rng = np.random.default_rng(29)
    first = rng.normal(size=(n, n))
    second = rng.normal(size=(n, n)) if cross else first
    spectrum = namespace["power_spectrum"]
    centers, actual = spectrum(first, second, nbins=7)
    frequency = np.fft.fftfreq(n, 1 / (n - 1))
    kk = np.hypot(*np.meshgrid(frequency, frequency, indexing="ij"))
    nyquist = (n - 1) / 2
    keep = (kk > 0) & (kk <= nyquist)
    edges = np.linspace(kk[keep].min(), nyquist, 8)
    counts = np.histogram(kk[keep], bins=edges)[0]
    weights = (np.fft.fft2(first) * np.fft.fft2(second).conj()).real
    full = np.histogram(kk[keep], bins=edges, weights=weights[keep])[0]
    ok = counts > 0
    np.testing.assert_allclose(centers, ((edges[:-1] + edges[1:]) / 2)[ok])
    np.testing.assert_allclose(actual, full[ok] / counts[ok], atol=1e-11)
    np.testing.assert_allclose(spectrum(first.T, second.T, nbins=7)[1], actual, atol=1e-11)
