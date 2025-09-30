# tests/_helpers.py
# type import
from typing import Any

# numerical import
import numpy as np

# pde import
import ufl

# dolfinx import
from dolfinx import fem

# local import
from forward_solver import SteadyHeat2DForwardSolver


def eval_obj(forward, T_obs, sigma: float, alpha: float) -> float:
    r"""
    Evaluate the objective function:
        J[T(h), h] = 0.5 * [ \int_\Omega [T(h) - T_obs]^2/\sigma^2 + \alpha * \int_\Omega (\nabla h)^2 ].
    See adjoint_solver.py and tao_solver.py for details.
    """
    dx = ufl.Measure("dx", domain=forward.mesh)
    T = forward.T
    h = forward.h.function
    delta_T = T - T_obs
    misfit = 0.5 * (1.0 / sigma**2) * ufl.inner(delta_T, delta_T) * dx
    reg = 0.5 * alpha * ufl.inner(ufl.grad(h), ufl.grad(h)) * dx
    return fem.assemble_scalar(fem.form(misfit + reg))


def pick_random_test_direction(V, seed=0, scale=1.0) -> fem.Function:
    r"""
    Pick a random direction \delta h in the function space V.
    """
    rng = np.random.default_rng(seed)
    d = fem.Function(V)
    d.x.array[:] = rng.normal(scale=scale, size=d.x.array.shape)
    return d


def h_true(x: Any) -> Any:
    """
    Return the true thermal conductivity.
    """
    return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)


def h0(x: Any) -> Any:
    """
    Return the fiducial/initial thermal conductivity.
    """
    return 2.0 + 3.0 * x[0] ** 2 + x[0] / (4.0 + 3.0 * x[1] ** 2)


def update_h(
    fwd: SteadyHeat2DForwardSolver, stepsize: float, delta_h: fem.Function
) -> None:
    """
    Update the thermal conductivity in the forward solver.
    """
    fwd.h.function.x.petsc_vec.axpy(stepsize, delta_h.x.petsc_vec)
    fwd.h.function.x.scatter_forward()
