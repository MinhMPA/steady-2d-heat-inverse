# tests/test_grad_taylorexp.py
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
from tests._helpers import eval_obj, pick_random_test_direction

n_mesh = 16
T_bottom = 300.0
noise_sigma = 1.0
reg_alpha = 5e-2

rnd_seed = 244
rnd_scale = 1.0
step_size = [1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1]

rtol, atol = 1e-5, 1e-8


def test_grad_taylorexp():
    r"""
    Examine the scaling of the Taylor expansion residuals with the step size. The residual is defined as:
        R(step_size) = J[h0 + step_size*\delta h] - J[h0] - step_size*(dJ/dh)_h0\cdot\delta h,
    where (dJ/dh)_h0 is the adjoint gradient evaluated at h0 and \delta h is a random test direction.
    The expected scaling is ln R(step_size) ~ 2 ln(step_size) for small step sizes.
    """

    # True h(x,y) used to generate observations
    def h_true(x):
        return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)

    # Initial guess for h(x,y) used in optimization
    def h0(x):
        return 2.0 + 3.0 * x[0] ** 2 + x[0] / (4.0 + 3.0 * x[1] ** 2)

    # Update h in the forward model
    def update_h(
        fwd: SteadyHeat2DForwardSolver, stepsize: float, delta_h: fem.Function
    ):
        fwd.h.function.x.petsc_vec.axpy(stepsize, delta_h.x.petsc_vec)
        fwd.h.function.x.scatter_forward()
        pass

    # Restore h in the forward model
    def restore_h(
        fwd: SteadyHeat2DForwardSolver, stepsize: float, delta_h: fem.Function
    ):
        fwd.h.function.x.petsc_vec.axpy(-stepsize, delta_h.x.petsc_vec)
        fwd.h.function.x.scatter_forward()
        pass

    # Compute convergence rate of Taylor expansion residuals
    def get_convergence_rate(R, step_size):
        R = np.asarray(R)
        step_size = np.asarray(step_size)
        conv_rate = np.log(R[:-1] / R[1:]) / np.log(step_size[:-1] / step_size[1:])
        return conv_rate

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

    ## Initial objective function
    J0 = eval_obj(fwd, T_obs, noise_sigma, reg_alpha)

    # Residual
    R = []

    # Pick random test direction
    delta_h = pick_random_test_direction(fwd.V, seed=rnd_seed, scale=rnd_scale)

    # Loop over step_size and compute Taylor expansion residuals
    for step in step_size:
        update_h(fwd, step, delta_h)
        ## New forward solution
        fwd.solve()
        ## New adjoint solution and gradient
        adj = SteadyHeat2DAdjointSolver(
            fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=T_bottom
        )
        adj.solve()
        adj.update_gradient()
        ## New objective function
        J = eval_obj(fwd, T_obs, noise_sigma, reg_alpha)
        ## Taylor expansion residual
        R.append(J - J0 - step * adj.grad.dot(delta_h.x.petsc_vec))
        restore_h(fwd, step, delta_h)
    cr = get_convergence_rate(R, step_size)
    print("Step sizes:", step_size)
    print("Residuals:", R)
    print("Convergence rates:", cr)
    assert np.allclose(
        cr, 2.0, rtol=rtol, atol=atol
    ), "Convergence rate is not close to 2 within target tolerance."
