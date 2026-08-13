# tests/test_tao_postconditions.py
# numerical imports
import sys

import numpy as np

sys.path.insert(0, "src")

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver


def _solved_pair():
    """A tiny converged forward/adjoint pair with a spatially varying h."""
    truth = SteadyHeat2DForwardSolver(nmesh=8, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0)
    truth.solve()
    T_obs = truth.add_noise(0.0, 1e-3, seed=0)

    fwd = SteadyHeat2DForwardSolver(nmesh=8, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0)
    fwd.solve()
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs.x.array, sigma=1e-3, alpha=5e-3, DBC_value=0.0
    )
    adj.solve()
    return fwd, adj


def test_shared_h_matches_returned_optimum_after_solve():
    """After solve(), the shared coefficient must be the optimum, not the last probe."""
    fwd, adj = _solved_pair()
    tao = SteadyHeat2DTAOSolver(fwd, adj, gatol=1e-8, grtol=1e-8, mit=50)

    sol = tao.solve()
    shared = fwd.h.function.x.array[: sol.size]

    np.testing.assert_allclose(shared, sol, rtol=1e-12, atol=0.0)


def test_solution_is_a_copy_not_a_live_view():
    """tao.solution must not alias TAO's internal vector."""
    fwd, adj = _solved_pair()
    tao = SteadyHeat2DTAOSolver(fwd, adj, gatol=1e-8, grtol=1e-8, mit=50)

    sol = tao.solve()
    before = sol.copy()
    tao.tao.getSolution().set(0.0)  # scribble on TAO's vector

    np.testing.assert_allclose(sol, before, rtol=0.0, atol=0.0)
