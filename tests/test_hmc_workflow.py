"""The documented heat experiment executes, preserving its saved physical fields."""

import os
from pathlib import Path
import sys

import pytest


import numpy as np

from heat_inference import HeatInverseProblem
from hmc_blackjax import require_sampling_environment
from posterior_samples import PosteriorSamples
from tests._hmc_helpers import make_problem, state_arrays
from tests.test_hmc_optimization import release


@pytest.fixture
def sampling_environment():
    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip(
            "Set S2DHI_HMC_QUALIFICATION=1 to execute the optional HMC notebook."
        )
    require_sampling_environment()
    __import__("arviz_stats")


def test_documented_sequence_preserves_fields_and_saved_results(
    tmp_path, sampling_environment
):
    problem = make_problem(nmesh=2)
    try:
        before = state_arrays(problem.forward)
        estimate = problem.optimize(initial_h=1.0)
        with problem.sample(
            initial=estimate,
            seed=7,
            chains=1,
            warmup=8,
            draws=12,
            output=tmp_path / "example",
        ) as samples:
            assert samples.load_draws().shape == (1, 12, 9)
            assert samples.summary().mean.shape == (9,)
            assert samples.diagnostics().qualified is False
            assert all(
                np.array_equal(a, b)
                for a, b in zip(before, state_arrays(problem.forward))
            )
            with HeatInverseProblem.load(samples.output) as loaded:
                assert loaded.to_record().problem_id == problem.to_record().problem_id
                with loaded.resume(samples.output) as resumed:
                    assert resumed.summary().counts.tolist() == [12]
    finally:
        release(problem)


def test_notebook_computation_cells_use_fine_data_and_coarse_inference(
    tmp_path, sampling_environment
):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/HMCSampling.ipynb", as_version=4)
    parameters = [
        cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameters) == 1
    # Every computation cell executes; physical rendering alone is excluded.
    parameters[0].source += (
        "\nnmesh = 4\nchains = 1\nwarmup = 8\ndraws = 12\n"
        f"output = {str(tmp_path / 'notebook-run')!r}\nplot = False\n"
    )
    manager = KernelManager(kernel_name="python3")
    # Use precisely the interpreter running this qualified physical calculation.
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=600,
        resources={"metadata": {"path": str(root)}},
        skip_cells_with_tag="plotting",
    )
    client.execute(cleanup_kc=True)
    executed = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(
        c.execution_count is not None
        for c in executed
        if "plotting" not in c.metadata.get("tags", [])
    )
    with PosteriorSamples.open(tmp_path / "notebook-run") as samples:
        assert samples.load_draws().shape == (1, 12, 25)
        assert samples.summary().counts.tolist() == [12]
        assert samples.diagnostics().qualified is False
        record = HeatInverseProblem.load(samples.output)
        try:
            assert record.to_record().arrays["sensor_points"].shape == (256, 2)
        finally:
            record.close()


def test_legacy_imports_do_not_load_optional_sampling_or_plotting():
    import subprocess

    code = """
import importlib.abc
import sys
class AbsentOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'blackjax', 'arviz_stats', 'pyvista', 'matplotlib'}:
            raise AssertionError('Eager optional import: ' + fullname)
sys.meta_path.insert(0, AbsentOptional())
import forward_solver, adjoint_solver, tao_solver, heat_inference, posterior_samples
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
