"""Exact continuation of the complete conductivity posterior calculation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
    pytest.skip("optional full-field sampling qualification", allow_module_level=True)

import numpy as np
from hmc_blackjax import HMCControls, require_sampling_environment
from posterior_store import RunStore
from ._hmc_helpers import make_problem, state_arrays, interrupt_after, assert_runs_equal

require_sampling_environment()


@pytest.fixture
def problem():
    from .test_hmc_optimization import release

    value = make_problem(nmesh=2)
    try:
        yield value
    finally:
        release(value)


@pytest.mark.parametrize("phase,completed", [("warmup", 3), ("sample", 4)])
def test_resume_matches_uninterrupted(problem, tmp_path, phase, completed):
    common = dict(initial=1.0, seed=42, chains=1, warmup=8, draws=12)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    before = state_arrays(problem.forward)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(
            output=tmp_path / "split",
            progress=interrupt_after(0, phase, completed),
            **common,
        )
    assert all(
        np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward))
    )
    with problem.resume(tmp_path / "split"):
        pass
    assert_runs_equal(tmp_path / "whole", tmp_path / "split")


def test_fresh_python_process_resumes_saved_heat_problem(problem, tmp_path):
    common = dict(initial=1.0, seed=142, chains=1, warmup=3, draws=4)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    output = tmp_path / "split"
    with pytest.raises(KeyboardInterrupt):
        problem.sample(
            output=output,
            progress=interrupt_after(0, "warmup", 1),
            **common,
        )
    script = """
import sys
from heat_inference import HeatInverseProblem

with HeatInverseProblem.load(sys.argv[1]) as problem:
    assert problem.forward is None
    with problem.resume(sys.argv[1]):
        pass
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert_runs_equal(tmp_path / "whole", output)


def test_extension_preserves_draws_and_initial_states(problem, tmp_path):
    output = tmp_path / "run"
    with problem.sample(
        initial=1.0, seed=3, warmup=2, draws=3, chains=2, output=output
    ):
        pass
    with RunStore.open(output) as store:
        old = [list(store.records(i)) for i in range(2)]
        starts = [store.initial_checkpoint(i) for i in range(2)]
    with problem.resume(output, total_draws=5) as result:
        assert result.output == output.resolve()
        assert result.metadata["total_draws"] == 5
    with RunStore.open(output) as store:
        from posterior_store import _same, _fields

        for i in range(2):
            assert _same(_fields(starts[i]), _fields(store.initial_checkpoint(i)))
            for a, b in zip(old[i], store.records(i)):
                assert _same(_fields(a), _fields(b))
    with pytest.raises(ValueError, match="reduce|smaller"):
        problem.resume(output, total_draws=4)


def test_zero_warmup_needs_explicit_spacing_and_disabled_adaptation(problem, tmp_path):
    for controls in (
        HMCControls(),
        HMCControls(adapt_step_size=False),
        HMCControls(initial_step_size=0.1),
    ):
        with pytest.raises(ValueError, match="warmup|explicit"):
            problem.sample(
                initial=1.0,
                seed=4,
                warmup=0,
                draws=1,
                chains=1,
                controls=controls,
                output=tmp_path / "invalid",
            )
        assert not (tmp_path / "invalid").exists()
    explicit = HMCControls(initial_step_size=0.1, adapt_step_size=False)
    with problem.sample(
        initial=1.0,
        seed=4,
        warmup=0,
        draws=1,
        chains=1,
        controls=explicit,
        output=tmp_path / "explicit",
    ) as result:
        assert result.metadata["controls"]["initial_step_size_explicit"] is True
    with problem.resume(tmp_path / "explicit", total_draws=2):
        pass


def test_default_four_chains_disperse_reproducibly(problem, tmp_path):
    common = dict(initial=1.0, seed=31, warmup=1, draws=1)
    for name in ("a", "b"):
        with problem.sample(output=tmp_path / name, **common):
            pass
    assert_runs_equal(tmp_path / "a", tmp_path / "b")
    with RunStore.open(tmp_path / "a") as store:
        assert store.metadata["chains"] == 4
        starts = [
            store.initial_checkpoint(i).inferred_coordinates["z"] for i in range(4)
        ]
        assert all(
            not np.array_equal(starts[i], starts[j]) for i in range(4) for j in range(i)
        )
        assert store.metadata["controls"]["initial_step_size_explicit"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"chains": 0},
        {"draws": True},
        {"warmup": -1},
    ],
)
def test_invalid_request_precedes_output(problem, tmp_path, options):
    common = dict(
        initial=1.0, seed=1, warmup=1, draws=1, chains=1, output=tmp_path / "invalid"
    )
    common.update(options)
    with pytest.raises(ValueError):
        problem.sample(**common)
    assert not (tmp_path / "invalid").exists()


def test_explicit_starts_and_single_chain_are_exact(problem, tmp_path):
    initial = problem.initial_state(1.0)
    with problem.sample(
        initial_states=[initial, initial],
        chains=2,
        seed=1,
        warmup=1,
        draws=1,
        output=tmp_path / "explicit",
    ):
        pass
    with problem.sample(
        initial=initial, chains=1, seed=1, warmup=1, draws=1, output=tmp_path / "single"
    ):
        pass
    for name, chains in (("explicit", 2), ("single", 1)):
        with RunStore.open(tmp_path / name) as store:
            for i in range(chains):
                assert np.array_equal(
                    store.initial_checkpoint(i).inferred_coordinates["z"],
                    initial.coordinates["z"],
                )


def test_changed_physics_or_environment_rejected_before_append(
    problem, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    with problem.sample(
        initial=1.0, seed=1, warmup=1, draws=1, chains=1, output=output
    ):
        pass
    with make_problem(nmesh=3) as other:
        with pytest.raises(ValueError, match="identity|contents"):
            other.resume(output, total_draws=2)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="environment"):
        problem.resume(output, total_draws=2)
    with RunStore.open(output) as store:
        assert store.metadata["total_draws"] == 1


def test_nonfinite_proposal_repeats_physical_fields(problem, tmp_path, monkeypatch):
    from conductivity_posterior import ConductivityPosterior, NonfiniteProposal

    original = ConductivityPosterior.evaluate
    calls = 0

    def reject(self, z):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise NonfiniteProposal("injected trial", np.asarray(z))
        return original(self, z)

    monkeypatch.setattr(ConductivityPosterior, "evaluate", reject)
    output = tmp_path / "repeats"
    with problem.sample(
        initial=1.0,
        seed=12,
        warmup=0,
        draws=3,
        chains=1,
        controls=HMCControls(initial_step_size=0.1, adapt_step_size=False),
        output=output,
    ):
        pass
    with RunStore.open(output) as store:
        start = store.initial_checkpoint(0)
        for record in store.records(0):
            assert np.array_equal(record.h, start.physical_state["h"])
            assert np.array_equal(
                record.sensor_temperature, start.physical_state["sensor_temperature"]
            )
            assert record.statistics["divergent"]
            assert not record.statistics["accepted"]
        assert store.checkpoint(0).counters["divergences"] == 3


@pytest.mark.parametrize("during_initialization", [False, True])
def test_failed_solve_replays_previous_boundary(
    problem, tmp_path, monkeypatch, during_initialization
):
    from conductivity_posterior import ConductivityPosterior, HeatSolveFailure

    original = ConductivityPosterior.evaluate
    calls = 0
    fail_at = 1 if during_initialization else 7

    def fail(self, z):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            # Count actual failed work as the physical evaluator would.
            self._heat_count += 1
            raise HeatSolveFailure("heat", z, -3, 1.0, 1)
        return original(self, z)

    common = dict(initial=1.0, seed=13, warmup=2, draws=3, chains=1)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    monkeypatch.setattr(ConductivityPosterior, "evaluate", fail)
    with pytest.raises(HeatSolveFailure):
        problem.sample(output=tmp_path / "split", **common)
    monkeypatch.setattr(ConductivityPosterior, "evaluate", original)
    with RunStore.open(tmp_path / "split") as store:
        failure = list(store.failures())[-1]
        assert failure["reason"] == -3
        assert failure["output"] == str((tmp_path / "split").resolve())
        assert failure["aborted_work"]["heat"] > 0
    with problem.resume(tmp_path / "split"):
        pass
    assert_runs_equal(tmp_path / "whole", tmp_path / "split")


def test_uneven_real_chains_resume_and_extend(problem, tmp_path):
    output = tmp_path / "uneven"
    common = dict(initial=1.0, seed=33, warmup=1, draws=3, chains=4)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(
            output=output, progress=interrupt_after(2, "sample", 2), **common
        )
    with RunStore.open(output) as store:
        assert [store.checkpoint(i).draws_done for i in range(4)] == [3, 3, 2, 0]
    with problem.resume(output):
        pass
    with problem.resume(output, total_draws=5):
        pass
    common["draws"] = 5
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    assert_runs_equal(tmp_path / "whole", output)


def test_uncommitted_interrupt_replays_identical_randomness(
    problem, tmp_path, monkeypatch
):
    from conductivity_posterior import ConductivityPosterior

    original = ConductivityPosterior.evaluate
    common = dict(initial=1.0, seed=111, warmup=3, draws=4, chains=1)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    count = 0

    def interrupt(self, z):
        nonlocal count
        count += 1
        result = original(self, z)
        if count == 7:
            raise KeyboardInterrupt("interrupted during Hamiltonian force")
        return result

    monkeypatch.setattr(ConductivityPosterior, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(output=tmp_path / "split", **common)
    monkeypatch.setattr(ConductivityPosterior, "evaluate", original)
    with RunStore.open(tmp_path / "split") as store:
        assert list(store.failures())[-1]["aborted_work"]["heat"] > 0
    with problem.resume(tmp_path / "split"):
        pass
    assert_runs_equal(tmp_path / "whole", tmp_path / "split")


@pytest.mark.parametrize(
    "change", ["observations", "prior", "source", "tolerances", "source_hash"]
)
def test_changed_identity_rejected_before_target_extension(
    problem, tmp_path, monkeypatch, change
):
    from heat_inference import HeatInverseProblem
    from conductivity_prior import GaussianLogConductivity
    from heat_model import HeatNumerics
    from temperature_observations import TemperatureObservations
    from ._hmc_helpers import make_forward, _release_problem

    output = tmp_path / "run"
    with problem.sample(
        initial=1.0, seed=1, warmup=1, draws=1, chains=1, output=output
    ):
        pass
    if change == "source_hash":
        import posterior_store

        original = posterior_store.numerical_environment

        def changed():
            record = original()
            record["sources"]["hmc_run"] = "modified"
            return record

        monkeypatch.setattr(posterior_store, "numerical_environment", changed)
        with pytest.raises(ValueError, match="sources|environment"):
            problem.resume(output, total_draws=2)
    else:
        obs = problem.posterior.observations
        prior = GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1)
        forward = make_forward(nmesh=2, q=2 if change == "source" else 1)
        if change == "observations":
            obs = TemperatureObservations(obs.points, obs.values + 0.01, obs.sigma)
        if change == "prior":
            prior = GaussianLogConductivity(mu=0.1, kappa=2, tau=1, h_ref=1)
        try:
            with HeatInverseProblem(
                forward=forward,
                observations=obs,
                prior=prior,
                numerics=HeatNumerics(rtol=1e-12) if change == "tolerances" else None,
            ) as other:
                with pytest.raises(ValueError, match="identity|contents"):
                    other.resume(output, total_draws=2)
        finally:
            _release_problem(forward.problem)
    with RunStore.open(output) as store:
        assert store.metadata["total_draws"] == 1
        assert store.checkpoint(0).draws_done == 1


@pytest.fixture
def analytic_transitions(problem, monkeypatch):
    """Manufactured independent Gaussian draws isolate long-run bookkeeping."""
    from types import SimpleNamespace
    import hmc_run
    from hmc_blackjax import BlackJAXChain, Transition
    from conductivity_posterior import SolveCounts

    class AnalyticPosterior:
        solve_counts = SolveCounts(0, 0, 0, 0)

        def evaluate(self, z):
            z = np.asarray(z).copy()
            return SimpleNamespace(
                z=z,
                potential=0.5 * z @ z,
                gradient=z,
                h=np.exp(z),
                sensor_temperature=np.full(9, 300.0),
            )

        def close(self):
            pass

    class AnalyticChain(BlackJAXChain):
        def transition(self, state, *, key, step_size, num_steps):
            import jax

            data = np.asarray(jax.random.key_data(key), dtype=np.uint32)
            rng = np.random.default_rng(data)
            z = rng.normal(size=np.shape(state.position))
            state = state._replace(
                position=z, logdensity=np.asarray(-0.5 * z @ z), logdensity_grad=-z
            )
            return Transition(state, 1.0, True, False, 0.0, 1.0, 1.0, num_steps, None)

    monkeypatch.setattr(problem, "_new_posterior", lambda **kwargs: AnalyticPosterior())
    monkeypatch.setattr(hmc_run, "BlackJAXChain", AnalyticChain)
    return problem


def test_accepted_uneven_2000_draw_calculation_extends_to_3000(
    analytic_transitions, tmp_path
):
    problem = analytic_transitions
    output = tmp_path / "uneven"
    controls = HMCControls(initial_step_size=0.1, adapt_step_size=False)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(
            initial=1.0,
            seed=17,
            warmup=0,
            draws=2000,
            chains=4,
            controls=controls,
            output=output,
            progress=interrupt_after(2, "sample", 1370),
        )
    with RunStore.open(output) as store:
        assert [store.checkpoint(i).draws_done for i in range(4)] == [
            2000,
            2000,
            1370,
            0,
        ]
        prefixes = [list(store.records(i)) for i in range(4)]
    with problem.resume(output):
        pass
    with RunStore.open(output) as store:
        assert [store.checkpoint(i).draws_done for i in range(4)] == [2000] * 4
    with problem.resume(output, total_draws=3000):
        pass
    from posterior_store import _same, _fields

    with RunStore.open(output) as store:
        assert [store.checkpoint(i).draws_done for i in range(4)] == [3000] * 4
        for i in range(4):
            assert all(
                _same(_fields(a), _fields(b))
                for a, b in zip(prefixes[i], store.records(i))
            )


def test_reordered_sequential_schedule_keeps_independent_histories(
    analytic_transitions, tmp_path, monkeypatch
):
    import builtins
    import hmc_run

    problem = analytic_transitions
    common = dict(initial=1.0, seed=21, warmup=2, draws=3, chains=4)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    # Interrupt after all starts exist; reverse only trajectory scheduling.
    with pytest.raises(KeyboardInterrupt):
        problem.sample(
            output=tmp_path / "reverse",
            progress=interrupt_after(0, "warmup", 1),
            **common,
        )
    monkeypatch.setattr(
        hmc_run, "range", lambda n: reversed(builtins.range(n)), raising=False
    )
    with problem.resume(tmp_path / "reverse"):
        pass
    assert_runs_equal(tmp_path / "whole", tmp_path / "reverse")


def test_half_integer_duration_rounds_up_and_saves_actual_duration(
    analytic_transitions, tmp_path, monkeypatch
):
    import jax

    monkeypatch.setattr(jax.random, "uniform", lambda *args, **kwargs: np.float64(1.25))
    controls = HMCControls(initial_step_size=0.1, adapt_step_size=False)
    with analytic_transitions.sample(
        initial=1.0,
        seed=1,
        warmup=0,
        draws=1,
        chains=1,
        controls=controls,
        output=tmp_path / "half",
    ):
        pass
    with RunStore.open(tmp_path / "half") as store:
        stats = next(store.records(0)).statistics
        assert stats["duration"] == 1.25
        assert stats["num_steps"] == 13
        assert stats["actual_duration"] == 13 * 0.1


@pytest.mark.parametrize("boundary", ["initialization", "trajectory"])
def test_interrupt_after_durable_commit_does_not_double_count(
    problem, tmp_path, monkeypatch, boundary
):
    original = RunStore._write_status
    common = dict(initial=1.0, seed=81, warmup=2, draws=3, chains=1)
    with problem.sample(output=tmp_path / "whole", **common):
        pass
    fired = False

    def interrupt_status(self):
        nonlocal fired
        original(self)
        try:
            cp = self.checkpoint(0)
        except KeyError:
            return
        at_boundary = cp.counters["trajectories"] == (
            0 if boundary == "initialization" else 1
        )
        if not fired and at_boundary:
            fired = True
            raise KeyboardInterrupt("after durable transaction before status returns")

    monkeypatch.setattr(RunStore, "_write_status", interrupt_status)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(output=tmp_path / "split", **common)
    monkeypatch.setattr(RunStore, "_write_status", original)
    with RunStore.open(tmp_path / "split") as store:
        assert all(
            value == 0 for value in list(store.failures())[-1]["aborted_work"].values()
        )
    with problem.resume(tmp_path / "split"):
        pass
    assert_runs_equal(tmp_path / "whole", tmp_path / "split")


def test_public_run_chains_rejects_changed_actual_initial_state(problem, tmp_path):
    from hmc_run import RunRequest, run_chains

    output = tmp_path / "saved"
    with problem.sample(
        initial=1.0, seed=1, warmup=1, draws=1, chains=1, output=output
    ):
        pass
    with RunStore.open(output, writable=True) as store:
        initial_states = [{"z": store.metadata["initial_states"][0]["z"] + 0.01}]
        with pytest.raises(ValueError, match="initial states"):
            run_chains(
                problem,
                initial_states=initial_states,
                request=RunRequest.from_metadata(store.metadata),
                store=store,
            )
        assert store.checkpoint(0).draws_done == 1
