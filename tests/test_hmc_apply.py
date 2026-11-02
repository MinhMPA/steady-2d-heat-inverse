"""Selected physical conductivities are applied reversibly to the same experiment."""

import numpy as np
import pytest
from dolfinx import fem

from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver
from forward_solver import SteadyHeat2DForwardSolver
from heat_model import FrozenHeatModel, coerce_field
from tests._hmc_helpers import make_forward, state_arrays, _release_problem
from tests.test_hmc_results import saved_known_run


def assert_unchanged(forward, before, handles):
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(forward)))
    assert forward.h.function is handles[0]
    assert forward.T is handles[1]
    assert forward.problem is handles[2]
    assert forward.a is handles[3] and forward.L is handles[4]
    assert forward.coefficient_generation == handles[5]


def handles(forward):
    return (
        forward.h.function,
        forward.T,
        forward.problem,
        forward.a,
        forward.L,
        forward.coefficient_generation,
    )


def test_apply_promotes_constant_and_rejects_changed_source(saved_known_run):
    samples = saved_known_run[0]
    forward = make_forward(nmesh=2, h=1.0)
    try:
        temperature = forward.T
        assert forward.coefficient_generation == 0
        result = samples.apply_to(forward, chain=0, draw=0)
        assert result is temperature and forward.T is temperature
        assert isinstance(forward.h.function, fem.Function)
        assert forward.coefficient_generation == 1
        assert np.array_equal(forward.problem.x.array, forward.T.x.array)
        assert forward.solve() is temperature
        assert forward.T is temperature
        assert np.array_equal(
            coerce_field(forward.h.function, FrozenHeatModel.from_forward(forward)),
            samples.field(chain=0, draw=0),
        )
        assert np.allclose(
            coerce_field(result, FrozenHeatModel.from_forward(forward)),
            samples.temperature(chain=0, draw=0),
            atol=2e-9,
        )
        forward.q.function.value = 9.0
        before, old = state_arrays(forward), handles(forward)
        with pytest.raises(ValueError, match="source"):
            samples.apply_to(forward, chain=0, draw=1)
        assert_unchanged(forward, before, old)
    finally:
        _release_problem(forward.problem)


@pytest.mark.parametrize("change,match", [("boundary", "boundary"), ("mesh", "mesh")])
def test_mismatched_experiment_leaves_fields_identical(saved_known_run, change, match):
    forward = make_forward(
        nmesh=3 if change == "mesh" else 2, bottom=299 if change == "boundary" else 300
    )
    try:
        forward.solve()
        before, old = state_arrays(forward), handles(forward)
        with pytest.raises(ValueError, match=match):
            saved_known_run[0].apply_to(forward, chain=0, draw=0)
        assert_unchanged(forward, before, old)
    finally:
        _release_problem(forward.problem)


def test_existing_field_handle_and_legacy_adjoint_remain_usable(saved_known_run):
    forward = make_forward(nmesh=2, h=lambda x: 2 + x[0])
    adjoint = None
    try:
        forward.solve()
        adjoint = SteadyHeat2DAdjointSolver(forward, forward.T, alpha=0.01)
        old = handles(forward)
        forward.h.function.x.array[:] = 5
        forward.T.x.array[:] = -100
        result = saved_known_run[0].apply_to(forward, summary="mean")
        assert result is old[1] and forward.h.function is old[0]
        assert forward.problem is old[2] and forward.coefficient_generation == old[5]
        adjoint.solve()
        gradient = adjoint.assemble_gradient()
        assert np.isfinite(gradient.array).all()
        gradient.destroy()
        tao = SteadyHeat2DTAOSolver(forward, adjoint, mit=1)
        try:
            g = tao.X0.duplicate()
            assert np.isfinite(tao._objgrad(tao.tao, tao.X0, g))
            g.destroy()
            assert np.isfinite(tao.solve()).all()
        finally:
            tao.tao.destroy()
    finally:
        if adjoint is not None:
            _release_problem(adjoint.problem)
        _release_problem(forward.problem)


def test_constant_promotion_rejects_stale_legacy_adjoint(saved_known_run):
    forward = make_forward(nmesh=2)
    adjoint = SteadyHeat2DAdjointSolver(forward, forward.T)
    try:
        saved_known_run[0].apply_to(forward, chain=0, draw=0)
        for operation in (adjoint.solve, adjoint.assemble_gradient):
            with pytest.raises(RuntimeError, match="[Rr]econstruct"):
                operation()
    finally:
        _release_problem(adjoint.problem)
        _release_problem(forward.problem)


def test_tao_rejects_generation_change_before_mutating_fields(saved_known_run):
    forward = make_forward(nmesh=2, h=lambda x: 1 + x[0])
    adjoint = SteadyHeat2DAdjointSolver(forward, forward.T)
    tao = SteadyHeat2DTAOSolver(forward, adjoint, mit=1)
    try:
        # Emulate an explicit coefficient replacement after this legacy object
        # was prepared; both public solve and callbacks must check first.
        forward._coefficient_generation += 1
        before = state_arrays(forward)
        gradient = tao.X0.duplicate()
        try:
            for operation in (
                tao.solve,
                lambda: tao._objgrad(tao.tao, tao.X0, gradient),
            ):
                with pytest.raises(RuntimeError, match="[Rr]econstruct"):
                    operation()
                assert all(
                    np.array_equal(a, b) for a, b in zip(before, state_arrays(forward))
                )
        finally:
            gradient.destroy()
    finally:
        tao.tao.destroy()
        _release_problem(adjoint.problem)
        _release_problem(forward.problem)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("initial", [1.0, lambda x: 1 + x[0]])
def test_failed_application_restores_handles_arrays_forms_and_generation(
    saved_known_run, monkeypatch, initial, error
):
    forward = make_forward(nmesh=2, h=initial)
    try:
        forward.solve()
        before, old = state_arrays(forward), handles(forward)
        original_solve = SteadyHeat2DForwardSolver.solve

        def fail(workspace):
            workspace.T.x.array[:] = -999
            if isinstance(workspace.h.function, fem.Function):
                workspace.h.function.x.array[:] = 99
            raise error("forced application failure")

        monkeypatch.setattr(SteadyHeat2DForwardSolver, "solve", fail)
        with pytest.raises(error, match="forced"):
            saved_known_run[0].apply_to(forward, chain=0, draw=2)
        assert_unchanged(forward, before, old)
        monkeypatch.setattr(SteadyHeat2DForwardSolver, "solve", original_solve)
        saved_known_run[0].apply_to(forward, chain=0, draw=2)
        assert np.isfinite(forward.T.x.array).all()
    finally:
        _release_problem(forward.problem)


@pytest.mark.parametrize("initial", [1.0, lambda x: 1 + x[0]])
def test_real_unconverged_application_rolls_back(saved_known_run, initial):
    forward = make_forward(nmesh=2, h=initial)
    try:
        forward.solve()
        forward.set_ksp_tolerances(rtol=1e-14, atol=1e-16, max_it=0)
        before, old = state_arrays(forward), handles(forward)
        policy = forward.problem.solver.getErrorIfNotConverged()
        with pytest.raises(RuntimeError, match="converg"):
            saved_known_run[0].apply_to(forward, chain=0, draw=2)
        assert_unchanged(forward, before, old)
        assert forward.problem.solver.getErrorIfNotConverged() == policy
    finally:
        _release_problem(forward.problem)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"chain": 0},
        {"draw": 0},
        {"chain": 0, "draw": 0, "summary": "mean"},
        {"summary": "nonsense"},
        {"summary": ("quantile", -1)},
    ],
)
def test_exactly_one_physical_selection_is_required(saved_known_run, kwargs):
    samples = saved_known_run[0]
    forward = make_forward(nmesh=2)
    try:
        before, old = state_arrays(forward), handles(forward)
        for operation in (
            lambda: samples.as_function(**kwargs),
            lambda: samples.apply_to(forward, **kwargs),
        ):
            with pytest.raises(ValueError):
                operation()
            assert_unchanged(forward, before, old)
    finally:
        _release_problem(forward.problem)


def test_sd_is_for_plotting_and_positive_quantile_is_applicable(saved_known_run):
    samples = saved_known_run[0]
    forward = make_forward(nmesh=2)
    try:
        before, old = state_arrays(forward), handles(forward)
        with pytest.raises(ValueError, match="standard deviation|sd"):
            samples.apply_to(forward, summary="sd")
        assert_unchanged(forward, before, old)
        samples.apply_to(forward, summary=("quantile", 0.5))
        assert np.allclose(
            coerce_field(forward.h.function, FrozenHeatModel.from_forward(forward)),
            samples.summary(quantiles=(0.5,)).quantiles[0],
        )
    finally:
        _release_problem(forward.problem)


@pytest.mark.parametrize(
    "values,coordinates",
    [
        ([-1] * 9, None),
        ([np.nan] * 9, None),
        ([1] * 8, None),
        ([1] * 9, np.zeros((9, 2))),
    ],
)
def test_direct_assignment_rejects_invalid_physical_fields(
    saved_known_run, values, coordinates
):
    samples = saved_known_run[0]
    forward = make_forward(nmesh=2)
    try:
        before, old = state_arrays(forward), handles(forward)
        with pytest.raises(ValueError):
            forward.apply_conductivity_field(
                np.array(values),
                coordinates=samples.coordinates if coordinates is None else coordinates,
            )
        assert_unchanged(forward, before, old)
    finally:
        _release_problem(forward.problem)


def test_prediction_and_plotting_failure_preserve_caller(saved_known_run, monkeypatch):
    from conductivity_posterior import ConductivityPosterior
    from heat_model import FrozenHeatModel

    samples = saved_known_run[0]
    forward = make_forward(nmesh=2)
    try:
        before = state_arrays(forward)

        def fail(*args, **kwargs):
            raise RuntimeError("private prediction failure")

        monkeypatch.setattr(ConductivityPosterior, "predict_h", fail)
        with pytest.raises(RuntimeError, match="private"):
            samples.temperature(chain=0, draw=0)
        monkeypatch.setattr(FrozenHeatModel, "_build_mesh", fail)
        with pytest.raises(RuntimeError, match="private"):
            samples.as_function(chain=0, draw=0)
        assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(forward)))
    finally:
        _release_problem(forward.problem)


def test_failed_constant_adoption_preserves_original_experiment(
    saved_known_run, monkeypatch
):
    import forward_solver

    forward = make_forward(nmesh=2)
    try:
        forward.solve()
        before, old = state_arrays(forward), handles(forward)
        original = forward_solver.LinearProblem
        calls = 0

        def fail_final_forms(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("forced adoption failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(forward_solver, "LinearProblem", fail_final_forms)
        with pytest.raises(RuntimeError, match="adoption"):
            saved_known_run[0].apply_to(forward, chain=0, draw=0)
        assert calls == 2
        assert_unchanged(forward, before, old)
    finally:
        _release_problem(forward.problem)
