"""Saved physical fields retain every state and its spatial association."""

from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from posterior_samples import PosteriorSamples
from posterior_store import ChainCheckpoint, RunStore, TrajectoryRecord
from tests._hmc_helpers import make_problem, _release_problem, state_arrays


def save_known_run(path, lengths=(3, 3), warmup=1, nmesh=2, cell="triangle"):
    """Public saved records from known whitened states on a real heat experiment."""
    problem = make_problem(nmesh=nmesh, cell=cell)
    fields, sensors, states = [], [], []
    try:
        n = problem.posterior.prior.dimension
        z_values = [np.linspace(-0.8, 0.4, n), np.linspace(0.2, 1.0, n)]
        evaluations = [problem.posterior.evaluate(z) for z in z_values]
        initial = []
        for chain in range(len(lengths)):
            result = evaluations[chain % 2]
            initial.append(
                ChainCheckpoint(
                    chain,
                    "warmup" if warmup else "sample",
                    0,
                    0,
                    {"z": z_values[chain % 2]},
                    {"h": result.h, "sensor_temperature": result.sensor_temperature},
                    {"position": z_values[chain % 2]},
                    {"key": np.array([chain, 7])},
                    {"step": 0},
                    0.05,
                    {"accepted": 0},
                )
            )
        with RunStore.create(
            path,
            problem=problem.to_record(),
            run_metadata={
                "total_draws": max(1, *lengths),
                "warmup": warmup,
                "chains": len(lengths),
            },
        ) as store:
            store.initialize_chains(initial)
            for chain, length in enumerate(lengths):
                cp = initial[chain]
                for index in range(warmup):
                    cp = replace(cp, warmup_done=index + 1)
                    store.commit_trajectory(
                        TrajectoryRecord(
                            chain,
                            "warmup",
                            index,
                            cp.inferred_coordinates,
                            cp.physical_state["h"],
                            cp.physical_state["sensor_temperature"],
                            {},
                        ),
                        cp,
                    )
                chain_h, chain_sensors, chain_z = [], [], []
                for draw in range(length):
                    # The second retained outcome repeats the first: a rejected move.
                    which = (chain + (draw == 2)) % 2
                    result, z = evaluations[which], z_values[which]
                    cp = replace(
                        cp,
                        phase="sample",
                        draws_done=draw + 1,
                        inferred_coordinates={"z": z},
                        hmc_state={"position": z},
                        physical_state={
                            "h": result.h,
                            "sensor_temperature": result.sensor_temperature,
                        },
                    )
                    store.commit_trajectory(
                        TrajectoryRecord(
                            chain,
                            "sample",
                            draw,
                            {"z": z},
                            result.h,
                            result.sensor_temperature,
                            {"accepted": draw != 1},
                        ),
                        cp,
                    )
                    chain_h.append(result.h.copy())
                    chain_sensors.append(result.sensor_temperature.copy())
                    chain_z.append(z.copy())
                fields.append(np.array(chain_h).reshape(length, n))
                sensors.append(np.array(chain_sensors).reshape(length, 9))
                states.append(chain_z)
        return fields, sensors, states
    finally:
        problem.close()
        _release_problem(problem.forward.problem)


@pytest.fixture
def saved_known_run(tmp_path):
    path = tmp_path / "known"
    fields, sensors, states = save_known_run(path)
    with PosteriorSamples.open(path) as samples:
        yield samples, np.array(fields), np.array(sensors), states


def test_saved_physical_summary_includes_rejection_repeats(
    saved_known_run, monkeypatch
):
    samples, expected, _, _ = saved_known_run
    monkeypatch.setattr(
        samples,
        "load_draws",
        lambda **kw: pytest.fail("summary loaded complete chains"),
    )
    result = samples.summary(node_chunk_size=2)
    assert np.allclose(result.mean, expected.mean(axis=(0, 1)))
    assert np.allclose(result.sd, expected.std(axis=(0, 1)))
    assert np.allclose(
        result.quantiles, np.quantile(expected, [0.05, 0.5, 0.95], axis=(0, 1))
    )
    assert not np.allclose(result.mean, np.exp(np.log(expected).mean(axis=(0, 1))))
    assert np.array_equal(result.probabilities, [0.05, 0.5, 0.95])
    assert np.array_equal(result.counts, [3, 3])
    assert np.array_equal(result.coordinates, samples.coordinates)
    assert not result.qualified


def test_draws_states_sensors_and_immutable_spatial_labels(saved_known_run):
    samples, h, sensors, states = saved_known_run
    assert np.array_equal(samples.load_draws(), h)
    assert np.array_equal(
        samples.load_draws(chains=[1, 0], start=1, stop=3), h[[1, 0], 1:3]
    )
    assert np.array_equal(samples.field(chain=0, draw=1), h[0, 1])
    assert np.array_equal(samples.sensor_predictions(chain=1, start=1), sensors[1, 1:])
    state = samples.state(chain=1, draw=2)
    assert np.array_equal(state.coordinates["z"], states[1][2])
    assert state.problem_id == samples._store.problem_record.problem_id
    state.coordinates["z"][:] = 999
    assert np.array_equal(samples.state(chain=1, draw=2).coordinates["z"], states[1][2])
    field = samples.field(chain=0, draw=0)
    field[:] = 99
    assert np.array_equal(samples.field(chain=0, draw=0), h[0, 0])
    for labels in (
        samples.coordinates,
        samples.cells,
        samples.summary().coordinates,
        samples.summary().cells,
    ):
        with pytest.raises(ValueError):
            labels.setflags(write=True)


def test_partial_chains_are_pooled_without_silent_truncation(tmp_path):
    path = tmp_path / "partial"
    fields, _, _ = save_known_run(path, lengths=(3, 1))
    with PosteriorSamples.open(path) as samples:
        summary = samples.summary(node_chunk_size=1)
        assert np.array_equal(summary.counts, [3, 1])
        assert not summary.qualified
        assert np.allclose(summary.mean, np.concatenate(fields).mean(axis=0))
        with pytest.raises(ValueError, match="equal|common"):
            samples.load_draws()
        assert samples.load_draws(start=0, stop=1).shape == (2, 1, 9)
        assert samples.load_draws(chains=[0]).shape == (1, 3, 9)


def test_no_retained_draws_has_explicit_error(tmp_path):
    path = tmp_path / "empty"
    save_known_run(path, lengths=(0, 0))
    with PosteriorSamples.open(path) as samples:
        with pytest.raises(ValueError, match="retained|draw"):
            samples.summary()
        assert samples.load_draws().shape == (2, 0, 9)
        assert samples.sensor_predictions(chain=0).shape == (0, 9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_chunk_size": 0},
        {"node_chunk_size": -1},
        {"node_chunk_size": True},
        {"node_chunk_size": 1.5},
        {"quantiles": [-0.01]},
        {"quantiles": [1.01]},
        {"quantiles": [np.nan]},
        {"quantiles": [[0.5]]},
    ],
)
def test_summary_rejects_invalid_selection(saved_known_run, kwargs):
    with pytest.raises(ValueError):
        saved_known_run[0].summary(**kwargs)


@pytest.mark.parametrize(
    "chain,draw", [(-1, 0), (2, 0), (0, -1), (0, 3), (True, 0), (0, 0.5)]
)
def test_selected_draw_is_range_checked(saved_known_run, chain, draw):
    with pytest.raises((ValueError, IndexError)):
        saved_known_run[0].field(chain=chain, draw=draw)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1},
        {"stop": 4},
        {"start": 2, "stop": 1},
        {"chains": []},
        {"chains": [0, 0]},
        {"chains": [2]},
    ],
)
def test_array_selection_is_explicit_and_range_checked(saved_known_run, kwargs):
    with pytest.raises((ValueError, IndexError)):
        saved_known_run[0].load_draws(**kwargs)


def test_private_temperature_matches_saved_sensors_and_preserves_caller(
    saved_known_run,
):
    from heat_inference import HeatInverseProblem
    from heat_model import coerce_field, FrozenHeatModel

    samples, _, sensors, _ = saved_known_run
    with HeatInverseProblem.load(samples.output) as problem:
        forward = problem._model.build_forward(
            prefix="result_expected_", numerics=problem._numerics
        )
        try:
            before = state_arrays(forward)
            temperature = samples.temperature(chain=0, draw=2)
            assert all(
                np.array_equal(a, b) for a, b in zip(before, state_arrays(forward))
            )
            forward.apply_conductivity_field(
                samples.field(chain=0, draw=2), coordinates=samples.coordinates
            )
            assert np.allclose(
                temperature,
                coerce_field(forward.T, FrozenHeatModel.from_forward(forward)),
                rtol=0,
                atol=2e-9,
            )
            assert np.array_equal(samples.sensor_predictions(chain=0)[2], sensors[0, 2])
        finally:
            _release_problem(forward.problem)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_plotting_function_keeps_linear_field_at_vtk_nodes(tmp_path, cell):
    from dolfinx.plot import vtk_mesh
    from heat_inference import HeatInverseProblem

    path = tmp_path / "linear"
    save_known_run(path, lengths=(1,), warmup=0, nmesh=3, cell=cell)
    # A public saved physical record can contain any positive field; use a linear
    # one to reveal coordinate permutations in the plotting conversion.
    with RunStore.open(path, writable=True) as store:
        xy = store.problem_record.arrays["coordinates"]
        h = 1 + 0.2 * xy[:, 0] + 0.3 * xy[:, 1]
        cp = store.checkpoint(0)
        with HeatInverseProblem.load(path) as problem:
            _, sensor = problem.posterior.predict_h(h)
            z = problem.posterior.prior.from_h(h)
        store.update_target(2)
        cp = replace(
            cp,
            draws_done=2,
            inferred_coordinates={"z": z},
            hmc_state={"position": z},
            physical_state={"h": h, "sensor_temperature": sensor},
        )
        store.commit_trajectory(
            TrajectoryRecord(0, "sample", 1, cp.inferred_coordinates, h, sensor, {}), cp
        )
    with PosteriorSamples.open(path) as samples:
        function = samples.as_function(chain=0, draw=1)
        _, _, coordinates = vtk_mesh(function.function_space)
        assert np.allclose(
            function.x.array, 1 + 0.2 * coordinates[:, 0] + 0.3 * coordinates[:, 1]
        )
        for selection in (
            {"summary": "mean"},
            {"summary": "sd"},
            {"summary": ("quantile", 0.5)},
        ):
            assert (
                samples.as_function(**selection).x.array.shape == function.x.array.shape
            )


def test_saved_summaries_import_without_sampling_or_heat_dependencies(saved_known_run):
    script = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax','blackjax','dolfinx','petsc4py','mpi4py'}:
            raise ImportError('blocked '+fullname)
sys.meta_path.insert(0, Block())
from posterior_samples import PosteriorSamples
with PosteriorSamples.open(sys.argv[1]) as samples:
    assert samples.summary(node_chunk_size=2).mean.shape == (9,)
    assert samples.load_draws().shape == (2,3,9)
    assert samples.field(chain=0,draw=0).shape == (9,)
    assert samples.sensor_predictions(chain=0).shape == (3,9)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(saved_known_run[0].output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
