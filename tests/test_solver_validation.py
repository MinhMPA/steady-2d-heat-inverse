# tests/test_solver_validation.py
# numerical imports
import pytest

# dolfinx imports
from dolfinx import fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver


def _solver_pair(h):
    """Build a tiny solved forward/adjoint pair for the given thermal conductivity `h`."""
    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=h, q=1.0, DBC_value=300.0)
    fwd.solve()
    T_obs = fem.Function(fwd.V)
    T_obs.x.array[:] = fwd.T.x.array
    T_obs.x.scatter_forward()
    adj = SteadyHeat2DAdjointSolver(fwd, T_obs, sigma=1.0, alpha=0.0, DBC_value=0.0)
    adj.solve()
    return fwd, adj


def test_constant_h_is_rejected_with_actionable_error():
    """A scalar h has no DOF vector for TAO to optimize -- say so, do not AttributeError."""
    fwd, adj = _solver_pair(4.0)

    with pytest.raises(TypeError, match="must be a fem.Function"):
        SteadyHeat2DTAOSolver(fwd, adj)


def test_callable_h_is_accepted():
    """A spatially-varying h yields a fem.Function and must construct cleanly."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    tao = SteadyHeat2DTAOSolver(fwd, adj)

    assert isinstance(fwd.h.function, fem.Function)
    assert tao.use_logh is True


def test_h_min_none_under_logh_raises_valueerror():
    """log(None) is meaningless -- reject it with the same ValueError as h_min <= 0."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    with pytest.raises(ValueError, match="h_min must be positive"):
        SteadyHeat2DTAOSolver(fwd, adj, h_min=None, use_logh=True)


def test_h_min_none_is_allowed_without_logh():
    """Optimizing directly in h, a None lower bound simply means 0.0."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    tao = SteadyHeat2DTAOSolver(fwd, adj, h_min=None, use_logh=False)

    assert tao.lb.min()[1] == pytest.approx(0.0)


def test_plotting_noisy_field_without_noise_points_at_add_noise():
    """The assertion must name the method that actually produces T_obs."""
    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0 + x[0], q=1.0)
    fwd.solve()

    with pytest.raises(AssertionError, match="add_noise"):
        fwd.plot_output_temperature(noiseless=False)
