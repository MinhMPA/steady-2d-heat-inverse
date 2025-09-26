# tests/_helpers.py
# numerical import
import numpy as np
# pde import
import ufl
# dolfinx import
from dolfinx import fem

def eval_cost(forward, T_obs, sigma: float, alpha: float) -> float:
    dx = ufl.Measure("dx", domain=forward.mesh)
    T = forward.T
    h = forward.h.function
    delta_T = T - T_obs
    misfit = 0.5 * (1.0 / sigma*sigma) * ufl.inner(delta_T, delta_T) * dx
    reg    = 0.5 * alpha * ufl.inner(ufl.grad(h), ufl.grad(h)) * dx
    return fem.assemble_scalar(fem.form(misfit + reg))

def rand_direction(V, seed=0, scale=1.0) -> fem.Function:
    rng = np.random.default_rng(seed)
    d = fem.Function(V)
    d.x.array[:] = rng.normal(scale=scale, size=d.x.array.shape)
    return d
