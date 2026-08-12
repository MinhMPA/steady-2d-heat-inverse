# tests/test_solver_validation.py
# numerical imports
import numpy as np
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
