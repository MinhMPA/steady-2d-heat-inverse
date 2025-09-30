# tests/test_grad_finitediff.py
import pytest

pytestmark = [pytest.mark.gradcheck]
# numerical import
import numpy as np, pytest, ufl

# pde import
from petsc4py import PETSc

# dolfinx import
from dolfinx import fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from ._helpers import eval_obj, pick_random_test_direction, h_true, h0, update_h

n_mesh = 16
T_bottom = 300.0
noise_sigma = 1.0
reg_alpha = 1e-6

rnd_seed = 244
rnd_scale = 1.0
step_size = 1e-5

rtol, atol = 1e-5, 1e-8


def test_grad_finitediff():
    r"""
    Compare adjoint and finite-difference directional derivatives:
        DJ[h0,\delta h]=dJ/dh\cdot\delta h.
    The finite-difference directional derivative is the central difference:
        DJ_fd[h0,\delta h] = (J(h0 + step_size*\delta h) - J(h0 - step_size*\delta_h)) / (2*step_size).
    """

    # True forward model
    fwd_truth = SteadyHeat2DForwardSolver(
        nmesh=n_mesh, mesh_type="quadrilateral", h=h_true, q=1.0, DBC_value=T_bottom
    )
    # True solution
    fwd_truth.solve()

    # Initial forward model
    fwd = SteadyHeat2DForwardSolver(
        nmesh=n_mesh, mesh_type="quadrilateral", h=h0, q=1.0, DBC_value=T_bottom
    )
    # Initial solution
    fwd.solve()

    # Observation = True solution (noise-free for this test)
    T_obs = fem.Function(fwd.V, name="T_obs")
    T_obs.x.array[:] = fwd_truth.T.x.array

    # Adjoint gradient
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=T_bottom
    )
    adj.solve()
    adj.update_gradient()

    # Finite-difference gradient
    ## Pick random test direction
    delta_h = pick_random_test_direction(fwd.V, seed=rnd_seed, scale=rnd_scale)
    ## Evaluate baseline objective at h0
    J0 = eval_obj(fwd, T_obs, noise_sigma, reg_alpha)
    ## Update h0 -> h0 + step_size*delta_h
    update_h(fwd, step_size, delta_h)
    ## Solve
    fwd.solve()
    ## Evaluate new objective
    J_plus = eval_obj(fwd, T_obs, noise_sigma, reg_alpha)
    ## Restore h0
    update_h(fwd, -step_size, delta_h)
    ## Update h0 -> h0 - step_size*delta_h
    update_h(fwd, -step_size, delta_h)
    ## Solve
    fwd.solve()
    ## Evaluate new objective
    J_minus = eval_obj(fwd, T_obs, noise_sigma, reg_alpha)

    # Assemble directional derivatives
    fd_dderiv = (J_plus - J_minus) / (2.0 * step_size)
    adj_dderiv = adj.grad.dot(delta_h.x.petsc_vec)

    # Error between adjoint and finite-difference gradients
    abs_err = abs(adj_dderiv - fd_dderiv)
    rel_err = abs_err / abs(fd_dderiv)
    assert abs_err <= atol + rtol * abs(
        fd_dderiv
    ), f"adj_dderiv={adj_dderiv}, fd_dderiv={fd_dderiv}, abs_err={abs_err}, rel_err={rel_err}."
