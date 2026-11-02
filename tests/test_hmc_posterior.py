"""Discrete sensor likelihood, Gaussian energy, and three force checks."""

from dataclasses import FrozenInstanceError
import subprocess
import sys
import textwrap
import numpy as np
import pytest
from conductivity_posterior import (
    ConductivityPosterior,
    HeatSolveFailure,
    NonfiniteProposal,
    SolveCounts,
)
from heat_model import FrozenHeatModel, HeatNumerics
from temperature_observations import TemperatureObservations
from ._hmc_helpers import (
    make_forward,
    make_posterior,
    sensor_tangent_derivative,
    state_arrays,
)

ACCURATE = HeatNumerics(rtol=1e-12, atol=1e-14, ksp_type="preonly", pc_type="lu")
STEPS = np.array([1e-2, 1e-3, 1e-4, 1e-5])


def test_posterior_contains_sensor_likelihood_and_one_gaussian_energy():
    with make_posterior() as posterior:
        z = np.linspace(-0.1, 0.1, posterior.prior.dimension)
        ev = posterior.evaluate(z)
        r = (
            ev.sensor_temperature - posterior.observations.values
        ) / posterior.observations.sigma
        assert ev.phi == pytest.approx(0.5 * r @ r)
        assert ev.potential == pytest.approx(ev.phi + 0.5 * z @ z)
        assert np.allclose(ev.gradient, ev.grad_phi + z, atol=1e-12)
        posterior.evaluate(-z)
        again = posterior.evaluate(z)
        assert np.array_equal(again.gradient, ev.gradient)
        assert again.potential == ev.potential
        assert posterior.solve_counts.heat == 3
        assert posterior.solve_counts.adjoint == 3
        assert posterior.solve_counts.heat_iterations >= ev.heat_iterations
        assert posterior.solve_counts.adjoint_iterations >= ev.adjoint_iterations
        for name in ("z", "h", "grad_phi", "gradient", "sensor_temperature"):
            assert getattr(ev, name).dtype == np.float64
        ev.z[:] = 99.0
        assert not np.array_equal(ev.z, z)
        assert np.array_equal(posterior.evaluate(z).z, z)
        with pytest.raises((AttributeError, FrozenInstanceError)):
            posterior.solve_counts.heat = 0
        with pytest.raises(AttributeError):
            posterior.prior = None
        with pytest.raises(AttributeError):
            posterior.observations = None


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_sensor_force_finite_difference_taylor_and_tangent(cell, heterogeneous):
    with make_posterior(cell=cell, numerics=ACCURATE) as initial:
        observations = initial.observations
        if heterogeneous:
            # Include constrained and insulated boundaries, and differing errors.
            points = np.vstack([observations.points, [[0.4, 0], [1, 0.7], [0.2, 1]]])
            observations = TemperatureObservations(
                points,
                np.r_[observations.values, [300.1, 300.2, 300.3]],
                np.linspace(0.035, 0.075, len(points)),
            )
        with ConductivityPosterior(
            initial._model,
            observations,
            initial.prior,
            ACCURATE,
            prefix=f"force_{cell}_{heterogeneous}_",
        ) as posterior:
            z = np.linspace(-0.2, 0.3, posterior.prior.dimension)
            direction = np.random.default_rng(244).normal(size=z.size)
            direction /= np.linalg.norm(direction)
            ev = posterior.evaluate(z)
            for value_name, gradient_name in (
                ("phi", "grad_phi"),
                ("potential", "gradient"),
            ):
                derivative = getattr(ev, gradient_name) @ direction
                differences, remainders = [], []
                for step in STEPS:
                    plus = getattr(posterior.evaluate(z + step * direction), value_name)
                    minus = getattr(
                        posterior.evaluate(z - step * direction), value_name
                    )
                    differences.append((plus - minus) / (2 * step))
                    remainders.append(
                        abs(plus - getattr(ev, value_name) - step * derivative)
                    )
                errors = np.abs(np.array(differences) - derivative) / abs(derivative)
                assert np.min(errors) < 1e-5, (
                    value_name,
                    differences,
                    derivative,
                    errors,
                )
                # Preselected interval 1e-2 -> 1e-3, before roundoff dominates.
                slope = np.log(remainders[0] / remainders[1]) / np.log(10.0)
                assert 1.8 < slope < 2.2, (value_name, remainders, slope)
            tangent = sensor_tangent_derivative(posterior, z, direction)
            assert abs(tangent - ev.grad_phi @ direction) / abs(tangent) < 1e-10
            posterior.evaluate(3 * direction)
            same = posterior.evaluate(z)
            assert same.potential == ev.potential
            assert np.array_equal(same.gradient, ev.gradient)


@pytest.mark.parametrize("numerics", [HeatNumerics(), ACCURATE])
def test_boundary_only_likelihood_has_no_conductivity_force(numerics):
    with make_posterior(numerics=numerics) as initial:
        obs = TemperatureObservations(
            [[0, 0], [0.4, 0], [1, 0]], [299, 301, 302], [1, 2, 3]
        )
        with ConductivityPosterior(
            initial._model, obs, initial.prior, numerics, prefix="boundary_only_"
        ) as posterior:
            ev = posterior.evaluate(np.zeros(posterior.prior.dimension))
            assert ev.phi == pytest.approx(0.5 * (1 + 1 / 4 + 4 / 9))
            assert np.max(np.abs(ev.grad_phi)) < 1e-12
            constrained = posterior._order[posterior._model.boundary_dofs]
            physical_temperature, _ = posterior.predict_h(ev.h)
            assert np.array_equal(
                physical_temperature[posterior._model.boundary_dofs],
                posterior._model.boundary_values,
            )
            assert np.array_equal(
                posterior._adjoint.x.array[constrained], np.zeros(len(constrained))
            )


def test_failed_heat_solve_is_not_a_nonfinite_proposal():
    posterior = make_posterior(
        numerics=HeatNumerics(rtol=1e-14, max_it=1, pc_type="none")
    )
    z = np.zeros(posterior.prior.dimension)
    with pytest.raises(HeatSolveFailure) as exc:
        posterior.evaluate(z)
    assert exc.value.stage == "heat"
    assert exc.value.iterations <= 1
    assert exc.value.reason < 0
    assert np.isfinite(exc.value.residual_norm)
    assert np.array_equal(exc.value.position, z)
    assert posterior.solve_counts.heat == 1
    assert posterior.solve_counts.adjoint == 0
    assert posterior.solve_counts.heat_iterations == exc.value.iterations
    with pytest.raises(RuntimeError, match="closed"):
        posterior.evaluate(z)
    posterior.close()


def test_failed_adjoint_retains_diagnostics_and_closes_owned_workspaces():
    posterior = make_posterior()
    posterior._adjoint_solver.setTolerances(rtol=1e-14, atol=0, max_it=1)
    posterior._adjoint_solver.getPC().setType("none")
    z = np.zeros(posterior.prior.dimension)
    with pytest.raises(HeatSolveFailure) as exc:
        posterior.evaluate(z)
    assert exc.value.stage == "adjoint"
    assert exc.value.reason < 0
    assert exc.value.iterations <= 1
    assert np.isfinite(exc.value.residual_norm)
    assert posterior.solve_counts.heat == 1
    assert posterior.solve_counts.adjoint == 1
    assert posterior.solve_counts.adjoint_iterations == exc.value.iterations
    with pytest.raises(RuntimeError, match="closed"):
        posterior.evaluate(z)
    posterior.close()


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e6, -1e6])
def test_nonfinite_or_unrepresentable_conductivity_recovers_without_solving(value):
    with make_posterior() as posterior:
        z = np.zeros(posterior.prior.dimension)
        ev = posterior.evaluate(z)
        counts = posterior.solve_counts
        with pytest.raises(NonfiniteProposal) as exc:
            posterior.evaluate(np.full(z.size, value))
        assert exc.value.stage in ("position", "conductivity")
        assert np.array_equal(
            exc.value.position, np.full(z.size, value), equal_nan=True
        )
        assert posterior.solve_counts == counts
        again = posterior.evaluate(z)
        assert again.potential == ev.potential
        assert np.array_equal(again.gradient, ev.gradient)


def test_predict_is_canonical_counts_only_heat_and_preserves_caller():
    forward = make_forward(h=lambda x: 1 + 0.1 * x[0])
    forward.solve()
    before = state_arrays(forward)
    model = FrozenHeatModel.from_forward(forward)
    with make_posterior() as initial:
        with ConductivityPosterior(
            model, initial.observations, initial.prior, HeatNumerics(), prefix="caller_"
        ) as posterior:
            z = np.zeros(posterior.prior.dimension)
            ev = posterior.evaluate(z)
            counts = posterior.solve_counts
            temperature, sensors = posterior.predict_h(ev.h)
            assert temperature.shape == z.shape
            assert np.array_equal(sensors, ev.sensor_temperature)
            assert np.array_equal(
                temperature[model.boundary_dofs], model.boundary_values
            )
            assert posterior.solve_counts.heat == counts.heat + 1
            assert posterior.solve_counts.adjoint == counts.adjoint
            assert np.all(np.diff(temperature.reshape(5, 5), axis=1) >= -1e-10)
    after = state_arrays(forward)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    posterior.close()
    with pytest.raises(RuntimeError, match="closed"):
        posterior.predict_h(ev.h)


def test_constructor_rejects_an_unbound_prior_and_bad_position_shape():
    from conductivity_prior import GaussianLogConductivity

    with make_posterior() as initial:
        with pytest.raises(TypeError, match="bound"):
            ConductivityPosterior(
                initial._model,
                initial.observations,
                GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1),
                HeatNumerics(),
                prefix="unbound_",
            )
        with pytest.raises(ValueError, match="shape"):
            initial.evaluate(np.zeros(initial.prior.dimension + 1))
        assert initial.solve_counts == SolveCounts(0, 0, 0, 0)


@pytest.mark.parametrize("stage", ["heat", "adjoint"])
def test_posterior_failure_preserves_legacy_divergence_reporting(stage):
    """Native PETSc divergence is tested in a child to contain its global error state."""
    code = textwrap.dedent("""
        import sys
        import numpy as np
        from petsc4py import PETSc
        from conductivity_posterior import HeatSolveFailure
        from heat_model import HeatNumerics
        from tests._hmc_helpers import make_posterior

        stage = sys.argv[1]
        if stage == "heat":
            posterior = make_posterior(numerics=HeatNumerics(
                rtol=1e-14, max_it=1, pc_type="none"))
        else:
            posterior = make_posterior()
            posterior._adjoint_solver.setTolerances(rtol=1e-14, atol=0, max_it=1)
            posterior._adjoint_solver.getPC().setType("none")
        try:
            posterior.evaluate(np.zeros(posterior.prior.dimension))
        except HeatSolveFailure as exc:
            assert exc.stage == stage
            assert exc.reason < 0
        else:
            raise AssertionError("The posterior must stop on an unconverged solve.")
        posterior.close()

        # Preserve the original forward solver's public configuration path.
        from forward_solver import SteadyHeat2DForwardSolver
        legacy = SteadyHeat2DForwardSolver(
            nmesh=8, h=lambda x: 1 + x[0], q=1,
            petsc_opts={"ksp_max_it": 1, "ksp_rtol": 1e-14, "pc_type": "none"})
        assert legacy.problem.solver.getErrorIfNotConverged()
        try:
            legacy.solve()
        except PETSc.Error:
            assert legacy.problem.solver.getConvergedReason() < 0
        else:
            raise AssertionError("The posterior failure suppressed legacy divergence.")
        print("LEGACY_DIVERGENCE_PRESERVED=" + stage)
        """)
    result = subprocess.run(
        [sys.executable, "-c", code, stage], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"LEGACY_DIVERGENCE_PRESERVED={stage}" in result.stdout


@pytest.mark.parametrize("numerics", [HeatNumerics(), ACCURATE])
def test_nonfinite_sensor_dual_is_recoverable_before_adjoint(numerics):
    # This artificial tiny-noise experiment needs an exactly representable
    # zero residual. A zero origin avoids rounding on public T = 300 + excess.
    with make_posterior(numerics=numerics, bottom=0.0) as initial:
        z = np.zeros(initial.prior.dimension)
        valid_z = np.full(z.size, 1e-8)
        baseline = initial.evaluate(z).sensor_temperature[4]
        valid_temperature = initial.evaluate(valid_z).sensor_temperature[4]
        residual = baseline - valid_temperature
        assert residual != 0
        # Each weight is ~0.75*float64_max, while the duplicate sum overflows.
        # The nearby reference state exactly matches these fixed observations.
        sigma = np.sqrt(abs(residual)) / np.sqrt(0.75 * np.finfo(np.float64).max)
        observations = TemperatureObservations(
            [[0.5, 0.5], [0.5, 0.5]], [valid_temperature, valid_temperature], sigma
        )
        standardized = residual / sigma
        weight = standardized / sigma
        assert np.isfinite(weight)
        assert np.isfinite(standardized**2)
        with ConductivityPosterior(
            initial._model,
            observations,
            initial.prior,
            numerics,
            prefix="nonfinite_sensor_dual_",
        ) as posterior:
            counts = posterior.solve_counts
            with pytest.raises(NonfiniteProposal) as exc:
                posterior.evaluate(z)
            assert exc.value.stage == "force"
            assert np.array_equal(exc.value.position, z)
            assert posterior.solve_counts.heat == counts.heat + 1
            assert posterior.solve_counts.adjoint == counts.adjoint
            valid = posterior.evaluate(valid_z)
            assert valid.phi == 0
            assert np.isfinite(valid.potential)
            assert np.isfinite(valid.gradient).all()
            assert posterior.solve_counts.adjoint == counts.adjoint + 1


@pytest.mark.parametrize("numerics", [HeatNumerics(rtol=1e-12, atol=1e-14), ACCURATE])
@pytest.mark.parametrize("spatial_boundary", [False, True])
def test_posterior_is_invariant_to_the_temperature_origin(numerics, spatial_boundary):
    """An arbitrary 300 K origin cannot change conductivity inference.

    Compare identical stored data and boundary differences at two origins,
    including a nonconstant imposed temperature. The 256 thermometers and
    1 mK errors expose absolute-temperature roundoff in the heat equation.
    No saved study or fitted state is needed for this physical symmetry test.
    """
    from contextlib import ExitStack
    from dataclasses import replace
    from conductivity_prior import GaussianLogConductivity
    from temperature_observations import SensorOperator
    from ._hmc_helpers import _release_problem

    caller = make_forward(nmesh=16, cell="quadrilateral", h=2.0, bottom=300.0)
    try:
        # A caller field is independent of this posterior's private heat solves.
        caller.T.x.array[:] = np.linspace(281.0, 317.0, len(caller.T.x.array))
        before = state_arrays(caller)
        model = FrozenHeatModel.from_forward(caller)
        if spatial_boundary:
            model = replace(
                model,
                boundary_values=300.0
                + 0.125 * model.coordinates[model.boundary_dofs, 0],
            )
        prior = GaussianLogConductivity(mu=np.log(2.0), kappa=2, tau=1, h_ref=1).bind(
            model, caller.V
        )
        axis = np.arange(1, 17) / 17.0
        points = np.array([(x, y) for x in axis for y in axis])
        values = 300.0 + (points[:, 1] - 0.5 * points[:, 1] ** 2) / 2.0
        values += np.random.default_rng(20261024).normal(0.0, 1e-3, len(points))
        observations = TemperatureObservations(points, values, 1e-3)
        shifted_model = replace(model, boundary_values=model.boundary_values - 300.0)
        shifted_observations = TemperatureObservations(points, values - 300.0, 1e-3)
        fingerprint = model.fingerprint
        saved_boundary = model.boundary_values.copy()
        saved_values = observations.values.copy()
        with ExitStack() as stack:
            physical = stack.enter_context(
                ConductivityPosterior(
                    model, observations, prior, numerics, prefix="origin_physical_"
                )
            )
            shifted = stack.enter_context(
                ConductivityPosterior(
                    shifted_model,
                    shifted_observations,
                    prior,
                    numerics,
                    prefix="origin_shifted_",
                )
            )
            z = np.linspace(-0.02, 0.02, prior.dimension)
            physical_ev, shifted_ev = physical.evaluate(z), shifted.evaluate(z)
            physical_T, physical_sensors = physical.predict_h(physical_ev.h)
            shifted_T, shifted_sensors = shifted.predict_h(shifted_ev.h)
            assert np.array_equal(physical_sensors, physical_ev.sensor_temperature)
            assert np.array_equal(shifted_sensors, shifted_ev.sensor_temperature)
            assert np.array_equal(physical_T[model.boundary_dofs], saved_boundary)
            # Public outputs retain physical temperature units.
            np.testing.assert_allclose(
                physical_T, shifted_T + 300.0, rtol=0, atol=2e-10
            )
            np.testing.assert_allclose(
                physical_sensors, shifted_sensors + 300.0, rtol=0, atol=2e-10
            )
            physical_native = np.empty_like(physical_T)
            physical_native[physical._order] = physical_T
            np.testing.assert_allclose(
                SensorOperator(physical._forward.V, points).predict(physical_native),
                physical_sensors,
                rtol=0,
                atol=2e-12,
            )
            # Both representations contain exactly the same centered doubles;
            # this checks the symmetry much more tightly than solver accuracy.
            assert physical_ev.phi == pytest.approx(
                shifted_ev.phi, rel=1e-12, abs=1e-10
            )
            assert physical_ev.potential == pytest.approx(
                shifted_ev.potential, rel=1e-12, abs=1e-10
            )
            np.testing.assert_allclose(
                physical_ev.grad_phi, shifted_ev.grad_phi, rtol=1e-12, atol=1e-9
            )
            np.testing.assert_allclose(
                physical_ev.gradient, shifted_ev.gradient, rtol=1e-12, atol=1e-9
            )
            assert physical._model is model
            assert physical.observations is observations
        assert model.fingerprint == fingerprint
        assert np.array_equal(model.boundary_values, saved_boundary)
        assert np.array_equal(observations.values, saved_values)
        assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(caller)))
    finally:
        _release_problem(caller.problem)
