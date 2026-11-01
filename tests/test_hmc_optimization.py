"""The posterior mode and complete physical-state initialization contract."""

from types import SimpleNamespace
import numpy as np
import pytest

from conductivity_posterior import HeatSolveFailure
from conductivity_prior import GaussianLogConductivity
from heat_inference import HeatInverseProblem, InferredState, PosteriorEstimate
from heat_model import HeatNumerics, NodalField
from temperature_observations import TemperatureObservations
from ._hmc_helpers import make_problem, make_forward, state_arrays, _release_problem


def release(problem):
    problem.close()
    if problem.forward is not None:
        _release_problem(problem.forward.problem)


@pytest.fixture
def problem():
    value = make_problem()
    try:
        yield value
    finally:
        release(value)


def test_optimization_handoff_preserves_posterior_and_caller(problem):
    before = state_arrays(problem.forward)
    result = problem.optimize(initial_h=1.0)
    assert isinstance(result, PosteriorEstimate)
    ev = problem.posterior.evaluate(result.state.coordinates["z"])
    assert result.objective == pytest.approx(ev.potential)
    assert result.success, result.message
    assert result.mode_coordinates == "log_conductivity"
    assert np.array_equal(result.conductivity, ev.h)
    assert problem.initial_state(result).problem_id == problem.problem_id
    assert all(
        np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward))
    )


def test_analytic_gaussian_mode_uses_full_potential(problem, monkeypatch):
    # Every coordinate has an independently known optimum and curvature.
    dimension = problem.posterior.prior.dimension
    optimum = np.linspace(-3, 4, dimension)
    curvature = np.linspace(1, 3, dimension)

    def gaussian(z):
        delta = z - optimum
        return SimpleNamespace(
            z=z.copy(),
            h=problem.posterior.prior.to_h(z),
            potential=7.0 + 0.5 * np.dot(curvature * delta, delta),
            gradient=curvature * delta,
        )

    monkeypatch.setattr(problem.posterior, "evaluate", gaussian)
    result = problem.optimize(initial_h=1.0, gtol=1e-9, ftol=1e-14)
    assert result.success, result.message
    assert np.allclose(result.state.coordinates["z"], optimum, atol=2e-7)
    assert result.objective == pytest.approx(7.0, abs=1e-12)
    assert np.allclose(result.conductivity, problem.posterior.prior.to_h(optimum))


def test_iteration_limit_is_an_explicit_failed_estimate(problem):
    result = problem.optimize(initial_h=0.3, maxiter=1, gtol=1e-14, ftol=1e-15)
    assert not result.success
    assert result.message
    assert np.isfinite(result.objective)
    assert (
        result.objective
        == problem.posterior.evaluate(result.state.coordinates["z"]).potential
    )
    assert problem.initial_state(result).problem_id == problem.problem_id


def test_nonfinite_trial_returns_last_finite_state_and_preserves_caller(
    problem, monkeypatch
):
    import heat_inference

    before = state_arrays(problem.forward)
    last = np.full(problem.posterior.prior.dimension, 0.25)
    expected = problem.posterior.evaluate(last)

    def trial(fun, x0, **kwargs):
        fun(x0)
        fun(last)
        fun(np.full_like(x0, 1e308))
        raise AssertionError("An unrepresentable trial must end optimization.")

    monkeypatch.setattr(heat_inference, "minimize", trial)
    result = problem.optimize(initial_h=1.0)
    assert not result.success
    assert "nonfinite" in result.message.lower()
    assert np.array_equal(result.state.coordinates["z"], last)
    assert np.array_equal(result.conductivity, expected.h)
    assert result.objective == expected.potential
    assert np.isfinite(problem.posterior.evaluate(last).potential)
    assert all(
        np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward))
    )


def test_optimizer_exception_preserves_caller(problem, monkeypatch):
    import heat_inference

    before = state_arrays(problem.forward)

    def failure(fun, x0, **kwargs):
        fun(x0)
        raise RuntimeError("external optimizer failure")

    monkeypatch.setattr(heat_inference, "minimize", failure)
    with pytest.raises(RuntimeError, match="external optimizer failure"):
        problem.optimize(initial_h=1.0)
    assert all(
        np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward))
    )


def test_failed_heat_solve_propagates_and_preserves_caller():
    source = make_problem()
    failed = None
    try:
        failed = HeatInverseProblem(
            forward=source.forward,
            observations=source.posterior.observations,
            prior=GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1),
            numerics=HeatNumerics(rtol=1e-14, max_it=1, pc_type="none"),
        )
        before = state_arrays(source.forward)
        with pytest.raises(HeatSolveFailure) as exc:
            failed.optimize(initial_h=1.0)
        assert exc.value.stage == "heat"
        assert exc.value.reason < 0
        assert all(
            np.array_equal(a, b) for a, b in zip(before, state_arrays(source.forward))
        )
    finally:
        if failed is not None:
            failed.close()
        release(source)


def state_only_problem(schema):
    """Local declared-coordinate fixture: no physical prior inverse or sampler."""
    value = object.__new__(HeatInverseProblem)
    value._closed = False
    value._problem_id = "declared-test-experiment"
    value._prior = SimpleNamespace(coordinate_schema=schema)
    return value


def test_complete_state_needs_no_inverse_and_keeps_auxiliary_fields():
    future = state_only_problem({"z": (9,), "interface": (1,)})
    state = InferredState(
        {"z": np.zeros(9), "interface": np.array([0.6])}, future.problem_id
    )
    carried = future.initial_state(state)
    assert np.array_equal(carried.coordinates["interface"], [0.6])
    assert np.array_equal(carried.coordinates["z"], np.zeros(9))
    assert carried.problem_id == future.problem_id
    carried.coordinates["interface"][0] = 0.2
    assert state.coordinates["interface"][0] == 0.6
    with pytest.raises(ValueError, match="inverse"):
        future.initial_state(1.0)


def test_supported_gaussian_state_needs_no_physical_inverse():
    value = state_only_problem({"z": (9,)})
    state = InferredState({"z": np.linspace(-1, 1, 9)}, value.problem_id)
    estimate = PosteriorEstimate(state, np.ones(9), 2.0, False, "iteration limit")
    assert np.array_equal(
        value.initial_state(estimate).coordinates["z"], state.coordinates["z"]
    )
    with pytest.raises(ValueError, match="inverse"):
        value.initial_state(1.0)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"z": np.zeros(8)},
        {"z": np.zeros(9), "unexpected": np.ones(1)},
        {},
        {"z": np.full(9, np.nan)},
        {"z": np.ones(9, dtype=complex) * 1j},
    ],
)
def test_initial_state_rejects_incomplete_or_invalid_coordinates(coordinates):
    value = state_only_problem({"z": (9,)})
    with pytest.raises(ValueError):
        value.initial_state(InferredState(coordinates, value.problem_id))


def test_initial_state_rejects_different_experiment(problem):
    state = InferredState(
        {"z": np.zeros(problem.posterior.prior.dimension)}, "another-experiment"
    )
    with pytest.raises(ValueError, match="problem|experiment"):
        problem.initial_state(state)


def test_physical_initialization_preserves_node_association(problem):
    model = problem._model
    h = 1 + 0.2 * model.coordinates[:, 0] + 0.1 * model.coordinates[:, 1]
    order = np.random.default_rng(9).permutation(len(h))
    field = NodalField(h[order], model.coordinates[order])
    z = problem.initial_state(field).coordinates["z"]
    assert np.allclose(problem.posterior.prior.to_h(z), h, atol=1e-13)
    with pytest.raises(ValueError, match="positive"):
        problem.initial_state(-1.0)


def test_frozen_experiment_and_independent_workspaces(problem):
    z = np.zeros(problem.posterior.prior.dimension)
    expected = problem.posterior.evaluate(z)
    identity = problem.problem_id
    problem.forward.q.function.value = 5.0
    problem.forward.T.x.array[:] = 0
    problem.forward.h.function.value = 3.0
    private = problem._new_posterior(prefix="separate_inference_test_")
    try:
        assert private is not problem.posterior
        assert private.evaluate(z).potential == expected.potential
        private.close()
        assert problem.posterior.evaluate(z).potential == expected.potential
        assert problem.problem_id == identity
        with pytest.raises(AttributeError):
            problem.problem_id = "modified"
    finally:
        private.close()


def test_problem_identity_tracks_physics_but_separates_numerics(problem):
    variants = []
    observations = problem.posterior.observations
    default = GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1)
    try:

        def build(obs=observations, prior=default, numerics=None, forward=None):
            item = HeatInverseProblem(
                forward=forward or problem.forward,
                observations=obs,
                prior=prior,
                numerics=numerics,
            )
            variants.append(item)
            return item

        same = build()
        accurate = build(numerics=HeatNumerics(rtol=1e-12))
        assert same.problem_id == problem.problem_id == accurate.problem_id
        assert same.numerics_id != accurate.numerics_id
        for changed in (
            TemperatureObservations(
                observations.points[::-1], observations.values[::-1], observations.sigma
            ),
            TemperatureObservations(
                observations.points, observations.values + 0.01, observations.sigma
            ),
            TemperatureObservations(observations.points, observations.values, 0.06),
        ):
            assert build(obs=changed).problem_id != problem.problem_id
        assert (
            build(
                prior=GaussianLogConductivity(mu=0.1, kappa=2, tau=1, h_ref=1)
            ).problem_id
            != problem.problem_id
        )
        source = make_forward(q=2)
        boundary = make_forward(bottom=299)
        mesh = make_forward(nmesh=3)
        try:
            for caller in (source, boundary, mesh):
                assert build(forward=caller).problem_id != problem.problem_id
        finally:
            for caller in (source, boundary, mesh):
                _release_problem(caller.problem)
    finally:
        for item in variants:
            item.close()


def test_context_releases_owned_workspaces_and_retains_caller(problem):
    forward = problem.forward
    private = problem._new_posterior(prefix="owned_extra_test_")
    with problem:
        pass
    problem.close()
    for posterior in (problem.posterior, private):
        with pytest.raises(RuntimeError, match="closed"):
            posterior.evaluate(np.zeros(posterior.prior.dimension))
    with pytest.raises(RuntimeError, match="closed"):
        problem.optimize(initial_h=1.0)
    forward.solve()
    assert np.isfinite(forward.T.x.array).all()


@pytest.mark.parametrize(
    "options",
    [
        {"maxiter": 0},
        {"maxiter": True},
        {"gtol": -1},
        {"gtol": np.nan},
        {"ftol": -1},
        {"ftol": np.inf},
    ],
)
def test_optimizer_rejects_invalid_stopping_parameters(problem, options):
    with pytest.raises(ValueError):
        problem.optimize(initial_h=1.0, **options)


def test_sampler_rejects_unsupported_coordinates_before_output(tmp_path):
    future = state_only_problem({"z": (9,), "interface": (1,)})
    complete = InferredState(
        {"z": np.zeros(9), "interface": np.array([0.6])}, future.problem_id
    )
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="unsupported.*coordinates"):
        future.sample(initial=complete, seed=1, output=output)
    assert not output.exists()
