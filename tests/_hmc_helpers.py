"""Small physical heat experiments for the full-field posterior tests."""

import numpy as np
from dolfinx import fem
from forward_solver import SteadyHeat2DForwardSolver


def make_forward(nmesh=4, cell="triangle", h=1.0, q=1.0, bottom=300.0):
    return SteadyHeat2DForwardSolver(
        nmesh=nmesh, mesh_type=cell, h=h, q=q, DBC_value=bottom
    )


def state_arrays(fwd):
    h = fwd.h.function
    values = h.x.array if isinstance(h, fem.Function) else np.asarray(h.value)
    return values.copy(), fwd.T.x.array.copy()


def _release_problem(problem):
    """Release LinearProblem ownership before its eventual Python finalizer."""
    for name in ("_solver", "_A", "_b", "_x", "_P_mat"):
        obj = getattr(problem, name, None)
        if obj is not None:
            obj.destroy()
            setattr(problem, name, None)


def make_posterior(nmesh=4, cell="triangle", numerics=None):
    """Fixed 3-by-3 thermometers and noiseless h=1+0.2*x+0.1*y data."""
    from conductivity_posterior import ConductivityPosterior
    from conductivity_prior import GaussianLogConductivity
    from heat_model import FrozenHeatModel, HeatNumerics
    from temperature_observations import SensorOperator, TemperatureObservations

    truth = make_forward(
        nmesh=nmesh, cell=cell, h=lambda x: 1 + 0.2 * x[0] + 0.1 * x[1]
    )
    try:
        truth.solve()
        axis = np.array([0.25, 0.5, 0.75])
        points = np.array([(x, y) for x in axis for y in axis])
        observations = TemperatureObservations(
            points, SensorOperator(truth.V, points).predict(truth.T.x.array), 0.05
        )
        model = FrozenHeatModel.from_forward(truth)
        prior = GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1).bind(
            model, truth.V
        )
    finally:
        _release_problem(truth.problem)
    return ConductivityPosterior(
        model, observations, prior, numerics or HeatNumerics(), prefix="hmc_test_"
    )


def sensor_tangent_derivative(posterior, z, direction):
    """Independent tangent heat equation followed by the sensor-space contraction."""
    import ufl
    from petsc4py import PETSc
    from dolfinx.fem.petsc import LinearProblem
    from heat_model import HeatNumerics, canonical_dof_indices
    from temperature_observations import SensorOperator

    numerics = HeatNumerics(rtol=1e-12, atol=1e-14, ksp_type="preonly", pc_type="lu")
    forward = posterior._model.build_forward(
        prefix="sensor_tangent_heat_", numerics=numerics
    )
    tangent = None
    try:
        order = canonical_dof_indices(forward.V)
        h = posterior.prior.to_h(z)
        forward.h.function.x.array[order] = h
        forward.h.function.x.scatter_forward()
        forward.solve()
        delta_h = fem.Function(forward.V)
        delta_log_h = posterior.prior.to_log_h(direction) - posterior.prior.to_log_h(
            np.zeros_like(z)
        )
        delta_h.x.array[order] = h * delta_log_h
        delta_h.x.scatter_forward()
        v = ufl.TestFunction(forward.V)
        rhs = -ufl.inner(delta_h * ufl.grad(forward.T), ufl.grad(v)) * ufl.dx
        bcs = [fem.dirichletbc(PETSc.ScalarType(0), forward.bottom_dofs, forward.V)]
        tangent = LinearProblem(
            forward.a,
            rhs,
            bcs=bcs,
            petsc_options_prefix="sensor_tangent_",
            petsc_options=numerics.petsc_options(),
        )
        dT = tangent.solve()
        sensor = SensorOperator(forward.V, posterior.observations.points)
        residual = sensor.predict(forward.T.x.array) - posterior.observations.values
        return float(
            (residual / posterior.observations.sigma**2) @ sensor.predict(dT.x.array)
        )
    finally:
        if tangent is not None:
            _release_problem(tangent)
        _release_problem(forward.problem)


def make_problem(nmesh=4, cell="triangle"):
    """The same fixed-sensor experiment with an accessible caller heat model."""
    from conductivity_prior import GaussianLogConductivity
    from heat_inference import HeatInverseProblem

    forward = make_forward(nmesh=nmesh, cell=cell)
    try:
        with make_posterior(nmesh=nmesh, cell=cell) as reference:
            return HeatInverseProblem(
                forward=forward,
                observations=reference.observations,
                prior=GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1),
            )
    except Exception:
        _release_problem(forward.problem)
        raise


def gaussian_value_and_grad(mean, covariance):
    """Exact Gaussian potential and force; likelihood subtracts ``0.5*z@z``.

    This is a manufactured probability law for Hamiltonian qualification. The
    returned full potential includes the unit Gaussian prior used by the split
    dynamics; its residual likelihood may have indefinite curvature.
    """
    mean = np.array(mean, dtype=np.float64, copy=True)
    covariance = np.array(covariance, dtype=np.float64, copy=True)
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Gaussian mean and covariance dimensions must agree.")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Gaussian parameters must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14):
        raise ValueError("Gaussian covariance must be symmetric.")
    np.linalg.cholesky(covariance)
    precision = np.linalg.solve(covariance, np.eye(len(mean)))

    def value_and_grad(z):
        displacement = z - mean
        gradient = precision @ displacement
        return np.float64(0.5 * displacement @ gradient), gradient

    return value_and_grad
