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
from forward_solver import SteadyHeatForwardSolver2D
from adjoint_solver import AdjointSteadyHeatSolver2D
from tests._helpers import eval_cost, rand_direction

T_bottom=300.0
noise_sigma=1.0
reg_alpha=1e-6

rnd_seed=244
rnd_scale=1.0
step_size=1e-6

rtol, atol = 1e-5, 1e-8

def test_adjoint_gradient_matches_fd():
    # Build "truth" to generate observations
    def h_true(x):
        return 1.0 + 6.0*x[0]**2 + x[0]/(1.0 + 2.0*x[1]**2)
    def h_init(x):
        return 2.0 + 3.0*x[0]**2 + x[0]/(4.0 + 3.0*x[1]**2)

    fwd_truth = SteadyHeatForwardSolver2D(nmesh=16, mesh_type="quadrilateral",
                                          h=h_true, q=1.0, DBC_value=T_bottom)
    fwd_truth.solve()

    # Optimization state starts from a constant h
    fwd = SteadyHeatForwardSolver2D(nmesh=16, mesh_type="quadrilateral",
                                    h=h_init, q=1.0, DBC_value=T_bottom)
    fwd.solve()

    # Observations = forward truth (noise-free for determinism)
    T_obs = fem.Function(fwd.V, name="T_obs")
    T_obs.x.array[:] = fwd_truth.T.x.array

    # Adjoint gradient
    adj = AdjointSteadyHeatSolver2D(fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=T_bottom)
    adj.solve()
    adj.update_gradient()
    g = adj.grad  # PETSc Vec representing dJ/dh in your chosen Riesz map

    # Finite-difference gradient
    ## Pick random test direction
    delta_h = rand_direction(fwd.V, seed=rnd_seed, scale=rnd_scale)

    ## Evaluate baseline objective at h0
    J0 = eval_cost(fwd, T_obs, noise_sigma, reg_alpha)

    ## Update h0 -> h0 + step_size*delta_h
    fwd.h.function.x.petsc_vec.axpy(step_size, delta_h.x.petsc_vec)
    fwd.h.function.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                           mode=PETSc.ScatterMode.FORWARD)
    ## Solve
    fwd.solve()
    ## Evaluate new objective
    J_plus = eval_cost(fwd, T_obs, noise_sigma, reg_alpha)

    fd_grad   = (J_plus - J0) / (step_size)
    adj_grad  = g.dot(delta_h.x.petsc_vec)

    abs_err = abs(adj_grad - fd_grad)
    rel_err = abs_err / abs(fd_grad)
    assert abs_err <= atol + rtol*abs(fd_grad), f"adj_grad={adj_grad}, fd_grad={fd_grad}, abs_err={abs_err}, rel_err={rel_err}."
