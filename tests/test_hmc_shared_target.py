"""The public supplied-adjoint connection and eager pair share one validator."""

import os
import pytest

if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
    pytest.skip(
        "Set S2DHI_HMC_QUALIFICATION=1 to require eager CPU float64 HMC qualification.",
        allow_module_level=True,
    )

from hmc_blackjax import require_sampling_environment

# Explicit qualification fails on missing/wrong packages instead of skipping.
require_sampling_environment()

import numpy as np
import jax
import jax.numpy as jnp
import hmc_blackjax as hmc
from conductivity_posterior import HeatSolveFailure, NonfiniteProposal
from ._hmc_helpers import make_posterior


def pair_call(route, target, x):
    logp, checked = target
    with jax.disable_jit():
        return checked(x) if route == "checked" else jax.value_and_grad(logp)(x)


@pytest.mark.parametrize("route", ["public", "checked"])
def test_single_call_copied_input_and_supplied_sign(route):
    calls = []

    def supplied(z):
        calls.append(z.copy())
        value, grad = np.float64(0.5 * z @ z), z.copy()
        z[:] = 99
        return value, grad

    target = hmc._make_validated_target(supplied)
    z = np.array([0.25, -0.5])
    value, grad = pair_call(route, target, z)
    assert len(calls) == 1 and np.array_equal(z, [0.25, -0.5])
    assert value == -0.15625 and np.array_equal(grad, [-0.25, 0.5])
    assert np.asarray(value).dtype == np.asarray(grad).dtype == np.float64


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize(
    "x,exception",
    [
        (np.array([1.0], dtype=np.float32), RuntimeError),
        (np.ones((1, 1)), ValueError),
        (np.array(1.0), ValueError),
    ],
)
def test_invalid_input_precedes_supplied_evaluation(route, x, exception):
    calls = []
    target = hmc._make_validated_target(lambda z: calls.append(z) or (np.float64(1), z))
    with pytest.raises(exception):
        pair_call(route, target, x)
    assert not calls


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize("transform", ["jit", "vmap", "pmap"])
def test_tracing_rejected_before_supplied_evaluation(route, transform):
    calls = []
    logp, checked = hmc._make_validated_target(
        lambda z: calls.append(z) or (np.float64(1), z)
    )
    fn = logp if route == "public" else checked
    x = jnp.array([0.2, -0.3], dtype=jnp.float64)
    with (
        jax.disable_jit(False),
        pytest.raises(RuntimeError, match="eager.*CPU|CPU.*eager"),
    ):
        getattr(jax, transform)(fn)(x if transform == "jit" else x[None, :])
    assert not calls


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize(
    "kind,exception",
    [
        ("value_shape", ValueError),
        ("force_shape", ValueError),
        ("value_dtype", RuntimeError),
        ("force_dtype", RuntimeError),
    ],
)
def test_returned_shape_and_dtype_checked_once(route, kind, exception):
    calls = []

    def supplied(z):
        calls.append(z)
        return {
            "value_shape": (np.ones(1), z),
            "force_shape": (np.float64(1), np.zeros(3)),
            "value_dtype": (np.float32(1), z),
            "force_dtype": (np.float64(1), z.astype(np.float32)),
        }[kind]

    with pytest.raises(exception):
        pair_call(route, hmc._make_validated_target(supplied), np.zeros(2))
    assert len(calls) == 1


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize("kind", ["heat", "adjoint", "nonfinite"])
@pytest.mark.parametrize("later", [False, True])
def test_exact_exception_passes_through_both_interfaces(route, kind, later):
    z = np.array([0.2, -0.3])
    sentinel = (
        NonfiniteProposal("force", z)
        if kind == "nonfinite"
        else HeatSolveFailure(kind, z, -3, 0.125, 2)
    )
    calls = 0

    def supplied(x):
        nonlocal calls
        calls += 1
        if calls == 1 + int(later):
            raise sentinel
        return np.float64(0.5 * x @ x), x.copy()

    target = hmc._make_validated_target(supplied)
    if later:
        pair_call(route, target, z)
    with pytest.raises(type(sentinel)) as exc:
        pair_call(route, target, z)
    assert exc.value is sentinel
    if kind == "nonfinite":
        value, force = pair_call(route, target, z)
        assert np.isfinite(value) and np.isfinite(force).all()


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize("stage", ["heat", "adjoint"])
def test_actual_failed_linear_solve_remains_fatal(route, stage):
    with make_posterior() as posterior:

        def supplied(z):
            ev = posterior.evaluate(z)
            return ev.potential, ev.gradient

        target = hmc._make_validated_target(supplied)
        z = np.zeros(posterior.prior.dimension)
        pair_call(route, target, z)
        solver = (
            posterior._forward.problem.solver
            if stage == "heat"
            else posterior._adjoint_solver
        )
        solver.setTolerances(rtol=1e-14, atol=0, max_it=1)
        solver.getPC().setType("none")
        with pytest.raises(HeatSolveFailure) as exc:
            pair_call(route, target, z)
        assert exc.value.stage == stage and exc.value.reason < 0
        with pytest.raises(RuntimeError, match="closed"):
            posterior.evaluate(z)


def test_actual_heat_supplied_pair_matches_public_and_direct_at_multiple_fields():
    with make_posterior() as posterior:

        def supplied(z):
            ev = posterior.evaluate(z)
            return ev.potential, ev.gradient

        target = hmc._make_validated_target(supplied)
        for z in (
            np.zeros(posterior.prior.dimension),
            np.linspace(-0.1, 0.1, posterior.prior.dimension),
        ):
            direct = posterior.evaluate(z)
            before = posterior.solve_counts
            checked = pair_call("checked", target, z)
            public = pair_call("public", target, z)
            assert checked[0] == public[0] == -direct.potential
            assert np.array_equal(checked[1], public[1]) and np.array_equal(
                checked[1], -direct.gradient
            )
            assert posterior.solve_counts.heat == before.heat + 2
            assert posterior.solve_counts.adjoint == before.adjoint + 2
        before = posterior.solve_counts
        for route in ("public", "checked"):
            with pytest.raises(NonfiniteProposal):
                pair_call(route, target, np.full_like(z, np.nan))
        assert posterior.solve_counts == before
        assert pair_call("checked", target, z)[0] == checked[0]


@pytest.mark.parametrize("route", ["public", "checked"])
@pytest.mark.parametrize("kind", ["value", "force", "exception", "heat", "adjoint"])
def test_midtrajectory_recovery_and_fatal_semantics(route, kind):
    calls = 0
    sentinel = (
        HeatSolveFailure(kind, np.zeros(2), -3, 0.125, 2)
        if kind in ("heat", "adjoint")
        else None
    )

    def supplied(z):
        nonlocal calls
        calls += 1
        if calls == 3:
            if sentinel is not None:
                raise sentinel
            if kind == "exception":
                raise NonfiniteProposal("force", z)
            if kind == "value":
                return np.float64(np.nan), z.copy()
            if kind == "force":
                return np.float64(0.5 * z @ z), np.full_like(z, np.inf)
        return np.float64(0.5 * z @ z), z.copy()

    chain = hmc.BlackJAXChain(
        supplied,
        hmc.HMCControls(initial_step_size=0.1, step_size_bounds=(1e-6, np.pi / 8)),
    )
    if route == "public":
        chain._checked_pair = jax.value_and_grad(chain.logdensity)
    state = chain.initialize(np.array([0.2, -0.3]))
    if sentinel is not None:
        with pytest.raises(HeatSolveFailure) as exc:
            chain.transition(state, key=jax.random.key(71), step_size=0.1, num_steps=4)
        assert exc.value is sentinel
    else:
        result = chain.transition(
            state, key=jax.random.key(71), step_size=0.1, num_steps=4
        )
        assert (
            not result.accepted
            and result.divergent
            and result.acceptance_probability == 0
        )
        assert np.array_equal(result.state.position, state.position)
        assert result.failure_reason
        result = chain.transition(
            state, key=jax.random.key(72), step_size=0.1, num_steps=4
        )
        assert np.isfinite(result.state.logdensity)
    assert calls >= 3
