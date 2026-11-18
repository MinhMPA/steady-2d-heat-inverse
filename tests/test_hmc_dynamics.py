"""Hamiltonian mechanics and heat-solve failure laws for the selected HMC route."""

import os

import pytest

if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
    pytest.skip(
        "Set S2DHI_HMC_QUALIFICATION=1 to require eager CPU float64 HMC qualification.",
        allow_module_level=True,
    )

from hmc_blackjax import require_sampling_environment

require_sampling_environment()

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from blackjax.mcmc import hmc
from blackjax.mcmc.integrators import IntegratorState
from blackjax.adaptation.step_size import dual_averaging_adaptation
from conductivity_posterior import HeatSolveFailure, NonfiniteProposal
from hmc_blackjax import (
    HMCControls,
    BlackJAXChain,
    gaussian_split_integrator,
    make_logdensity,
    encode_state,
    decode_state,
    encode_adaptation,
    decode_adaptation,
)
from ._hmc_helpers import gaussian_value_and_grad, make_posterior

MEAN = np.array([0.3, -0.4, 0.2])
COV = np.array([[0.9, 0.24, -0.15], [0.24, 0.7, 0.1], [-0.15, 0.1, 1.2]])


def integrator_state(logp, z, p):
    z = jnp.asarray(z, dtype=jnp.float64)
    with jax.disable_jit():
        value, grad = jax.value_and_grad(logp)(z)
    return IntegratorState(z, jnp.asarray(p, dtype=jnp.float64), value, grad)


def energy(state):
    return float(-state.logdensity + 0.5 * state.momentum @ state.momentum)


def integrate(step, state, eps, count):
    with jax.disable_jit():
        for _ in range(count):
            state = step(state, eps)
    return state


def assert_same_state(left, right):
    assert left._fields == right._fields
    for a, b in zip(left, right):
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert np.asarray(a).shape == np.asarray(b).shape
        assert np.asarray(a).dtype == np.asarray(b).dtype


@pytest.mark.parametrize("momentum", [[1.0, 0.4], [-0.7, 1.2], [0.0, 0.0]])
def test_prior_only_rotation_conserves_energy_and_reverses(momentum):
    logp = make_logdensity(lambda z: (np.float64(0.5 * z @ z), z.copy()))
    step = gaussian_split_integrator(logp, lambda p: 0.5 * jnp.dot(p, p))
    start = integrator_state(logp, [0.2, -0.3], momentum)
    end = integrate(step, start, 0.2, 13)
    back = integrate(step, end, -0.2, 13)
    assert np.allclose(back.position, start.position, atol=2e-14, rtol=0)
    assert np.allclose(back.momentum, start.momentum, atol=2e-14, rtol=0)
    assert abs(energy(end) - energy(start)) < 2e-14
    # With zero likelihood force the full trajectory is one analytic rotation.
    assert np.allclose(
        end.position,
        np.cos(2.6) * start.position + np.sin(2.6) * start.momentum,
        atol=2e-14,
        rtol=0,
    )


@pytest.mark.parametrize(
    "momentum", [[1.0, 0.4, -0.5], [-0.7, 1.2, 0.1], [0.0, 0.0, 0.0]]
)
def test_correlated_gaussian_reversibility_and_second_order_energy(momentum):
    logp = make_logdensity(gaussian_value_and_grad(MEAN, COV))
    step = gaussian_split_integrator(logp, lambda p: 0.5 * jnp.dot(p, p))
    start = integrator_state(logp, [0.2, -0.3, 0.5], momentum)
    errors = []
    for eps in (0.1, 0.05, 0.025):
        count = round(1.0 / eps)
        end = integrate(step, start, eps, count)
        back = integrate(step, end, -eps, count)
        assert np.max(np.abs(back.position - start.position)) < 1e-10
        assert np.max(np.abs(back.momentum - start.momentum)) < 1e-10
        errors.append(abs(energy(end) - energy(start)))
    assert np.all(
        (np.array(errors[:-1]) / errors[1:] > 3.7)
        & (np.array(errors[:-1]) / errors[1:] < 4.3)
    ), errors


def test_phase_space_volume_is_preserved():
    logp = make_logdensity(gaussian_value_and_grad(MEAN, COV))
    step = gaussian_split_integrator(logp, lambda p: 0.5 * jnp.dot(p, p))
    x = np.array([0.2, -0.3, 0.5, 1.0, 0.4, -0.5])

    def evolve(y):
        end = integrate(step, integrator_state(logp, y[:3], y[3:]), 0.1, 7)
        return np.concatenate((end.position, end.momentum))

    delta = 1e-5
    jacobian = np.column_stack(
        [
            (evolve(x + delta * e) - evolve(x - delta * e)) / (2 * delta)
            for e in np.eye(6)
        ]
    )
    assert abs(np.linalg.det(jacobian) - 1) < 1e-6


def test_prior_only_transition_accepts_and_energy_uses_sampled_momentum():
    chain = BlackJAXChain(
        gaussian_value_and_grad(np.zeros(3), np.eye(3)),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.initialize(np.array([0.2, -0.3, 0.5]))
    key = jax.random.key(7501)
    transition = chain.transition(state, key=key, step_size=0.1, num_steps=12)
    with jax.disable_jit():
        expected, info = hmc.build_kernel(integrator=gaussian_split_integrator)(
            key, state, chain.logdensity, 0.1, jnp.ones(3), 12
        )
    assert_same_state(transition.state, expected)
    assert transition.accepted and not transition.divergent
    assert abs(transition.acceptance_probability - 1) < 1e-12
    assert transition.initial_energy == pytest.approx(
        float(-state.logdensity + 0.5 * info.momentum @ info.momentum), abs=1e-14
    )
    assert transition.final_energy == float(info.energy)
    assert (
        transition.delta_energy == transition.final_energy - transition.initial_energy
    )
    assert transition.failure_reason is None and transition.num_steps == 12


def test_ordinary_metropolis_rejection_repeats_library_state():
    target = gaussian_value_and_grad(np.zeros(2), 0.01 * np.eye(2))
    chain = BlackJAXChain(
        target,
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            divergence_threshold=0.001,
        ),
    )
    state = chain.initialize(np.zeros(2))
    key = jax.random.key(517)
    transition = chain.transition(state, key=key, step_size=0.35, num_steps=3)
    with jax.disable_jit():
        expected, info = hmc.build_kernel(
            integrator=gaussian_split_integrator, divergence_threshold=0.001
        )(key, state, chain.logdensity, 0.35, jnp.ones(2), 3)
    assert not transition.accepted
    assert transition.acceptance_probability == float(info.acceptance_rate)
    assert transition.divergent and transition.failure_reason is None
    assert_same_state(transition.state, state)
    assert_same_state(transition.state, expected)
    assert transition.state is not state


@pytest.mark.parametrize(
    "failure", ["exception", "value", "force", "momentum_overflow"]
)
def test_nonfinite_proposal_explicitly_repeats_previous_state(failure):
    calls = 0

    def supplied(z):
        nonlocal calls
        calls += 1
        if calls == 3:
            if failure == "exception":
                raise NonfiniteProposal("conductivity", z)
            if failure == "value":
                return np.float64(np.nan), z.copy()
            if failure == "force":
                return np.float64(0.5 * z @ z), np.full_like(z, np.inf)
            return np.float64(0.5 * z @ z), np.full_like(z, 1e308)
        return np.float64(0.5 * z @ z), z.copy()

    chain = BlackJAXChain(
        supplied,
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.initialize(np.array([0.2, -0.3]))
    first = chain.transition(state, key=jax.random.key(71), step_size=0.1, num_steps=4)
    assert not first.accepted and first.divergent and first.acceptance_probability == 0
    assert first.failure_reason
    assert first.initial_energy is not None and np.isfinite(first.initial_energy)
    assert first.final_energy is None and first.delta_energy is None
    assert_same_state(first.state, state)
    again = chain.transition(state, key=jax.random.key(72), step_size=0.1, num_steps=2)
    assert again.accepted and not again.divergent


def test_real_heat_posterior_reverses_and_uses_one_adjoint_per_step():
    with make_posterior(nmesh=3) as posterior:

        def supplied(z):
            ev = posterior.evaluate(z)
            return ev.potential, ev.gradient

        chain = BlackJAXChain(
            supplied,
            HMCControls(
                initial_step_size=0.1,
                step_size_bounds=(1e-6, np.pi / 8),
                initial_spread=0.3,
            ),
        )
        z = np.linspace(-0.02, 0.02, posterior.prior.dimension)
        p = np.linspace(0.1, -0.1, len(z))
        start = integrator_state(chain.logdensity, z, p)
        counts = posterior.solve_counts
        step = gaussian_split_integrator(
            chain.logdensity, lambda p: 0.5 * jnp.dot(p, p)
        )
        end = integrate(step, start, 0.001, 4)
        back = integrate(step, end, -0.001, 4)
        assert np.max(np.abs(back.position - start.position)) < 1e-10
        assert np.max(np.abs(back.momentum - start.momentum)) < 1e-10
        assert posterior.solve_counts.heat - counts.heat == 8
        assert posterior.solve_counts.adjoint - counts.adjoint == 8
        transition = chain.transition(
            chain.initialize(z), key=jax.random.key(62), step_size=0.001, num_steps=4
        )
        assert np.isfinite(transition.initial_energy) and np.isfinite(
            transition.final_energy
        )


@pytest.mark.parametrize("stage", ["heat", "adjoint"])
def test_mid_trajectory_real_pde_failure_propagates(stage):
    with make_posterior(nmesh=3) as posterior:
        calls = 0

        def supplied(z):
            nonlocal calls
            calls += 1
            if calls == 3:
                solver = (
                    posterior._forward.problem.solver
                    if stage == "heat"
                    else posterior._adjoint_solver
                )
                solver.setTolerances(rtol=1e-14, atol=0, max_it=1)
                solver.getPC().setType("none")
            ev = posterior.evaluate(z)
            return ev.potential, ev.gradient

        chain = BlackJAXChain(
            supplied,
            HMCControls(
                initial_step_size=0.1,
                step_size_bounds=(1e-6, np.pi / 8),
                initial_spread=0.3,
            ),
        )
        state = chain.initialize(np.zeros(posterior.prior.dimension))
        with pytest.raises(HeatSolveFailure) as caught:
            chain.transition(
                state, key=jax.random.key(79), step_size=0.001, num_steps=5
            )
        assert calls == 3 and caught.value.stage == stage
        assert caught.value.reason < 0


def test_controls_are_frozen_and_own_normalized_intervals():
    interval, bounds = [0.8, 2.2], [1e-6, 0.3]
    controls = HMCControls(trajectory_interval=interval, step_size_bounds=bounds)
    interval[0], bounds[1] = 0, 2
    assert controls.trajectory_interval == (0.8, 2.2)
    assert controls.step_size_bounds == (1e-6, 0.3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        controls.initial_step_size = 0.2


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_step_size": 0},
        {"initial_step_size": np.nan},
        {"initial_step_size": 0.5},
        {"target_acceptance": 1},
        {"target_acceptance": 0},
        {"trajectory_interval": (0, 2)},
        {"trajectory_interval": (2, 1)},
        {"trajectory_interval": (1, np.pi)},
        {
            "trajectory_interval": (0.2, 2),
            "initial_step_size": 0.1,
            "step_size_bounds": (1e-6, np.pi / 8),
        },
        {"step_size_bounds": (0, 0.1)},
        {"step_size_bounds": (0.2, 0.1)},
        {"divergence_threshold": np.inf},
        {"divergence_threshold": 0},
        {"initial_spread": -0.1},
        {"adapt_step_size": "yes"},
    ],
)
def test_controls_reject_invalid_or_resonant_choices(changes):
    with pytest.raises((ValueError, TypeError)):
        HMCControls(**changes)


def test_dual_averaging_matches_pinned_routine_and_caps_size():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    init, update, final = dual_averaging_adaptation(0.8, t0=10, gamma=0.05, kappa=0.75)
    with jax.disable_jit():
        actual, expected = chain.begin_warmup(), init(0.1)
        assert_same_state(actual, expected)
        for probability in [0.7, 0.8, 0.9, 1.0, 1.0, 0.5]:
            expected = update(expected, probability)
            expected = expected._replace(
                log_step_size=jnp.minimum(expected.log_step_size, np.log(np.pi / 8))
            )
            actual = chain.adapt(actual, probability)
            assert_same_state(actual, expected)
        assert chain.final_step_size(actual) == min(float(final(expected)), np.pi / 8)
    manual = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            adapt_step_size=False,
            initial_step_size=0.2,
            step_size_bounds=(1e-6, np.pi / 8),
        ),
    )
    initial = manual.begin_warmup()
    assert_same_state(manual.adapt(initial, 0.1), initial)
    assert manual.final_step_size(initial) == 0.2
    assert manual.current_step_size(initial) == 0.2


def test_dual_averaging_reports_below_bound_with_evidence():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.begin_warmup()
    with pytest.raises(ValueError, match="lower bound"):
        for _ in range(100):
            state = chain.adapt(state, 0.0)
    with pytest.raises(ValueError):
        chain.adapt(chain.begin_warmup(), np.nan)


def test_complete_named_state_roundtrips_preserve_values_shapes_dtypes_and_copies():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.initialize(MEAN)
    adaptation = chain.adapt(chain.begin_warmup(), 0.7)
    for original, encode, decode in [
        (state, encode_state, decode_state),
        (adaptation, encode_adaptation, decode_adaptation),
    ]:
        fields = encode(original)
        assert set(fields) == set(original._fields)
        restored = decode(fields)
        assert_same_state(original, restored)
        for name in fields:
            if isinstance(fields[name], np.ndarray):
                fields[name][...] = 999
        assert_same_state(original, restored)
        with pytest.raises(ValueError):
            decode({**encode(original), "unexpected_field": 1})
        with pytest.raises(ValueError):
            decode({})


def test_negative_large_energy_error_is_diagnostic_and_keeps_metropolis_acceptance():
    target = gaussian_value_and_grad(np.zeros(2), 0.01 * np.eye(2))
    chain = BlackJAXChain(
        target,
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            divergence_threshold=0.001,
        ),
    )
    state = chain.initialize(np.array([10.0, 10.0]))
    transition = chain.transition(
        state, key=jax.random.key(149), step_size=0.05, num_steps=1
    )
    assert transition.delta_energy < -0.001
    assert transition.divergent and transition.accepted
    assert transition.acceptance_probability == 1
    assert transition.failure_reason is None


def test_unavailable_initial_energy_is_null_and_never_a_finite_sentinel():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.initialize(MEAN)._replace(logdensity=jnp.asarray(-np.inf))
    transition = chain.transition(
        state, key=jax.random.key(23), step_size=0.1, num_steps=2
    )
    assert transition.initial_energy is None
    assert transition.final_energy is None and transition.delta_energy is None
    assert not transition.accepted and transition.divergent
    assert transition.failure_reason


def test_saved_hamiltonian_and_adaptation_continue_identically():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.initialize(MEAN)
    adaptation = chain.begin_warmup()
    # Include the initial library state whose step/averaged error fields are
    # Python scalars, then the updated mixture of scalars and zero-d arrays.
    for _ in range(3):
        fields = encode_adaptation(adaptation)
        restored = decode_adaptation(fields)
        assert_same_state(adaptation, restored)
        assert_same_state(chain.adapt(adaptation, 0.7), chain.adapt(restored, 0.7))
        adaptation = chain.adapt(adaptation, 0.7)
    next_key = jax.random.key(9471)
    original = chain.transition(state, key=next_key, step_size=0.1, num_steps=8)
    resumed = chain.transition(
        decode_state(encode_state(state)), key=next_key, step_size=0.1, num_steps=8
    )
    assert_same_state(original.state, resumed.state)
    assert original.acceptance_probability == resumed.acceptance_probability
    assert original.initial_energy == resumed.initial_energy
    assert original.final_energy == resumed.final_energy
    assert original.accepted == resumed.accepted


def test_finite_enormous_dual_averaging_proposal_caps_before_exponentiation():
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    # For a prior-only/easy likelihood target with acceptance one, the pinned
    # recursion has avg_error_n = (target-1)*n/(n+t0). Repeated upper capping
    # leaves mu=log(10*eps_initial)=0 and yields a finite raw log proposal
    # near 1265 at step 100000. No substituted adaptation routine is used.
    upper = chain.controls.step_size_bounds[1]
    state = chain.begin_warmup()._replace(
        step=100000,
        avg_error=-0.2 * 99999 / (99999 + 10),
        log_step_size=jnp.asarray(np.log(upper)),
        log_step_size_avg=jnp.asarray(np.log(upper)),
    )
    _, pinned_update, _ = dual_averaging_adaptation(0.8, t0=10, gamma=0.05, kappa=0.75)
    with jax.disable_jit():
        raw = pinned_update(state, 1.0)
    assert np.isfinite(raw.log_step_size)
    assert float(raw.log_step_size) > np.log(np.finfo(np.float64).max)
    updated = chain.adapt(state, 1.0)
    assert float(updated.log_step_size) == np.log(upper)
    assert float(updated.mu) == 0.0
    assert updated.step == state.step + 1


def test_finite_enormous_averaged_spacing_caps_and_tiny_average_fails_with_evidence():
    # Synthetic restored-state robustness: ordinary warmup averages already
    # capped current spacings and cannot reach a log average of 1000.
    chain = BlackJAXChain(
        gaussian_value_and_grad(MEAN, COV),
        HMCControls(
            initial_step_size=0.1,
            step_size_bounds=(1e-6, np.pi / 8),
            initial_spread=0.3,
        ),
    )
    state = chain.begin_warmup()._replace(log_step_size_avg=jnp.asarray(1000.0))
    assert chain.final_step_size(state) == chain.controls.step_size_bounds[1]
    lower = chain.controls.step_size_bounds[0]
    state = state._replace(log_step_size_avg=jnp.asarray(np.log(lower) - 1))
    with pytest.raises(ValueError, match="lower bound") as caught:
        chain.final_step_size(state)
    assert repr(lower) in str(caught.value)


@pytest.mark.parametrize(
    "bounds,initial",
    [
        ((1e-6, 0.1), 0.1),
        ((0.1, 0.2), 0.1),
        ((0.1, 0.1), 0.1),
        ((0.3, 0.3), 0.3),
        ((0.1, np.nextafter(0.1, np.inf)), 0.1),
        ((1e-200, 1e-199), 1e-200),
    ],
)
def test_current_final_and_transition_share_exact_recorded_spacing_bounds(
    bounds, initial
):
    chain = BlackJAXChain(
        gaussian_value_and_grad(np.zeros(2), np.eye(2)),
        HMCControls(initial_step_size=initial, step_size_bounds=bounds),
    )
    state = chain.initialize(np.array([0.2, -0.3]))
    adaptation = chain.adapt(chain.begin_warmup(), 1.0)
    # Reproduce the .1 upper cap through the physical transition first: the
    # stored log field need not exponentiate to an exactly representable bound.
    raw_spacing = float(jnp.exp(adaptation.log_step_size))
    chain.transition(state, key=jax.random.key(81), step_size=raw_spacing, num_steps=2)
    for boundary in bounds:
        at_boundary = adaptation._replace(
            log_step_size=jnp.asarray(np.log(boundary)),
            log_step_size_avg=jnp.asarray(np.log(boundary)),
        )
        for spacing in (
            chain.current_step_size(at_boundary),
            chain.final_step_size(at_boundary),
        ):
            assert bounds[0] <= spacing <= bounds[1]
            chain.transition(
                state, key=jax.random.key(82), step_size=spacing, num_steps=2
            )
    assert bounds[0] <= chain.current_step_size(adaptation) <= bounds[1]


@pytest.mark.parametrize(
    "bounds", [(1e-6, 0.1), (0.1, 0.1), (0.1, np.nextafter(0.1, np.inf))]
)
def test_transition_snaps_only_roundoff_excursions_to_actual_boundaries(bounds):
    chain = BlackJAXChain(
        gaussian_value_and_grad(np.zeros(2), np.eye(2)),
        HMCControls(initial_step_size=bounds[1], step_size_bounds=bounds),
    )
    state = chain.initialize(np.array([0.2, -0.3]))
    for offered, canonical in [
        (np.nextafter(bounds[0], 0), bounds[0]),
        (np.nextafter(bounds[1], np.inf), bounds[1]),
    ]:
        key = jax.random.key(999)
        actual = chain.transition(state, key=key, step_size=offered, num_steps=3)
        expected = chain.transition(state, key=key, step_size=canonical, num_steps=3)
        assert_same_state(actual.state, expected.state)
        assert actual.initial_energy == expected.initial_energy
        assert actual.final_energy == expected.final_energy
        assert actual.acceptance_probability == expected.acceptance_probability
    for offered in [0.0, -bounds[0], bounds[0] * (1 - 1e-8), bounds[1] * (1 + 1e-8)]:
        with pytest.raises(ValueError):
            chain.transition(
                state, key=jax.random.key(999), step_size=offered, num_steps=3
            )
    lower = bounds[0]
    genuinely_too_small = chain.begin_warmup()._replace(
        log_step_size=jnp.asarray(np.log(lower) - 1e-8),
        log_step_size_avg=jnp.asarray(np.log(lower) - 1e-8),
    )
    with pytest.raises(ValueError, match="lower bound"):
        chain.current_step_size(genuinely_too_small)
    with pytest.raises(ValueError, match="lower bound"):
        chain.final_step_size(genuinely_too_small)
