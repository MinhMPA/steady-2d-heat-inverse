"""Reference-informed numerical controls, with explicit provenance preserved."""

import os

import numpy as np
import pytest
from hmc_blackjax import HMCControls


def test_reference_defaults_preserve_duration_and_explicit_spacing_marker():
    controls = HMCControls()
    assert controls.initial_step_size == 6.5e-4
    assert controls.step_size_bounds == (1e-6, 1.3e-3)
    assert controls.initial_spread == 1.0
    assert not controls.initial_step_size_explicit
    assert controls.trajectory_interval == (np.pi / 4, 3 * np.pi / 4)
    assert controls.target_acceptance == 0.8
    assert controls.adapt_step_size
    assert HMCControls(initial_step_size=6.5e-4).initial_step_size_explicit


def test_ordinary_sampling_records_reference_defaults_and_matches_explicit(tmp_path):
    if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
        pytest.skip("optional full-field sampling qualification")
    from hmc_blackjax import require_sampling_environment

    require_sampling_environment()
    from ._hmc_helpers import make_problem, assert_runs_equal, state_arrays
    from .test_hmc_optimization import release
    from posterior_store import RunStore

    # RED checks the intended settings before costly physical trajectories.
    assert HMCControls().initial_step_size == 6.5e-4
    problem = make_problem(nmesh=2)
    before = state_arrays(problem.forward)
    explicit = HMCControls(
        initial_step_size=6.5e-4, step_size_bounds=(1e-6, 1.3e-3), initial_spread=1.0
    )
    try:
        for name, controls in (("ordinary", None), ("explicit", explicit)):
            with problem.sample(
                initial=1.0,
                seed=31,
                warmup=1,
                draws=1,
                controls=controls,
                output=tmp_path / name,
            ) as samples:
                assert samples.metadata["chains"] == 4
                assert not samples.diagnostics().qualified
        assert_runs_equal(tmp_path / "ordinary", tmp_path / "explicit")
        with (
            RunStore.open(tmp_path / "ordinary") as ordinary,
            RunStore.open(tmp_path / "explicit") as supplied,
        ):
            left, right = dict(ordinary.metadata["controls"]), dict(
                supplied.metadata["controls"]
            )
            assert left.pop("initial_step_size_explicit") is False
            assert right.pop("initial_step_size_explicit") is True
            assert left == right
            assert left["initial_step_size"] == 6.5e-4
            assert tuple(left["step_size_bounds"]) == (1e-6, 1.3e-3)
            assert left["initial_spread"] == 1.0
            for chain in range(4):
                assert ordinary.initial_checkpoint(chain).step_size == 6.5e-4
                for row in ordinary.records(chain):
                    assert 1e-6 <= row.statistics["step_size"] <= 1.3e-3
                    assert np.pi / 4 <= row.statistics["duration"] <= 3 * np.pi / 4
        assert all(
            np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward))
        )
    finally:
        release(problem)


def test_reference_default_does_not_supply_explicit_zero_warmup_spacing():
    from hmc_run import RunRequest

    for controls in (HMCControls(), HMCControls(adapt_step_size=False)):
        with pytest.raises(ValueError, match="explicit"):
            RunRequest(warmup=0, draws=1, chains=1, seed=1, controls=controls)
    request = RunRequest(
        warmup=0,
        draws=1,
        chains=1,
        seed=1,
        controls=HMCControls(initial_step_size=6.5e-4, adapt_step_size=False),
    )
    assert request.metadata()["controls"]["initial_step_size_explicit"] is True
