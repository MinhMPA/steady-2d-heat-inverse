import pytest

# tests/test_grad_forwarddiff.py

pytestmark = [pytest.mark.gradcheck]
# numerical import
import numpy as np, pytest, ufl

# pde import
from petsc4py import PETSc

# dolfinx import
from dolfinx import mesh, fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from _tangent_solver import _SteadyHeat2DTangentSolver
from tests._helpers import pick_random_test_direction

n_mesh = 16
T_bottom = 300.0
noise_sigma = 1.0
reg_alpha = 1e-6

rnd_seed = 244
rnd_scale = 1.0

rtol, atol = 1e-5, 1e-8


def test_grad_forwarddiff():
    r"""
    Compare adjoint directional derivative to forward directional derivative

    The forward directional derivative is computed by first solving the tangent equation for \delta T:
        \nabla\cdot(h\nabla\delta T) = -\nabla\cdot(\delta h\nabla T),
    then assembling the directional derivative as:
        (dJ/dh)\cdot\delta h = (1/\sigma^2)\int_\Omega dx [(T-T_obs)\cdot\delta T + \alpha(\nabla h\cdot\nabla\delta h)].
    """

    # True h(x,y) used to generate observations
    def h_true(x):
        return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)

    # Initial guess for h(x,y) used in optimization
    def h_init(x):
        return 2.0 + 3.0 * x[0] ** 2 + x[0] / (4.0 + 3.0 * x[1] ** 2)

    # True forward model
    fwd_truth = SteadyHeat2DForwardSolver(
        nmesh=n_mesh, mesh_type="quadrilateral", h=h_true, q=1.0, DBC_value=T_bottom
    )
    # True solution
    fwd_truth.solve()

    # Initial forward model
    fwd = SteadyHeat2DForwardSolver(
        nmesh=n_mesh, mesh_type="quadrilateral", h=h_init, q=1.0, DBC_value=T_bottom
    )
    # Initial solution
    fwd.solve()

    # Observation = True solution (noise-free for this test)
    T_obs = fem.Function(fwd.V, name="T_obs")
    T_obs.x.array[:] = fwd_truth.T.x.array

    # Pick random test direction for h
    delta_h = pick_random_test_direction(fwd.V, seed=rnd_seed, scale=rnd_scale)

    # Forward tangential directional derivative
    ## Solve tangent equation for dT
    tangent = _SteadyHeat2DTangentSolver(fwd)
    dT = tangent.assemble_jvp(delta_h)
    dx = ufl.Measure("dx", domain=fwd.mesh)
    # Misfit term
    misfit_deriv = (1.0 / noise_sigma**2) * ufl.inner(fwd.T - T_obs, dT) * dx
    ## Regularization term
    reg_deriv = reg_alpha * ufl.inner(ufl.grad(fwd.h.function), ufl.grad(delta_h)) * dx
    ## Assemble forward tangential directional derivative
    fwd_dir_deriv = fem.assemble_scalar(fem.form(misfit_deriv + reg_deriv))

    # Adjoint directional derivative
    ## Solve adjoint equation for lambda and g=dJ/dh
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=0.0
    )
    adj.solve()
    adj.update_gradient()
    ## Assemble adjoint directional derivatives
    adj_dir_deriv = adj.grad.dot(delta_h.x.petsc_vec)

    # ============ Compare Results ============
    # Error between adjoint and forward directional derivatives
    abs_err = abs(adj_dir_deriv - fwd_dir_deriv)
    rel_err = abs_err / abs(fwd_dir_deriv)

    assert abs_err <= atol + rtol * abs(
        fwd_dir_deriv
    ), f"adj_dir_deriv={adj_dir_deriv}, fwd_dir_deriv={fwd_dir_deriv}, abs_err={abs_err}, rel_err={rel_err}."
