# Testing

Gradient correctness is the thing worth testing here: an adjoint gradient that is subtly
wrong still *converges*, just to the wrong answer. It is therefore verified three
independent ways, all marked with the `gradcheck` marker.

```bash
pytest -m gradcheck
```

The tests run real FEM solves on a $16\times16$ mesh and take minutes, not seconds.

## The three checks

| Test | Compares the adjoint gradient against |
|---|---|
| `tests/test_grad_finitediff.py` | a central finite-difference directional derivative |
| `tests/test_grad_forwarddiff.py` | a tangent-linear (JVP) solve from `tests/_tangent_solver.py` |
| `tests/test_grad_taylorexp.py` | a Taylor remainder, whose convergence rate must be ≈ 2 |

They are genuinely independent: finite differences probe the objective, the tangent-linear
solve probes the *linearized* PDE, and the Taylor remainder probes the *order* of the
approximation. A sign error, a missing chain-rule factor, and a wrong regularization term
fail different subsets of the three — which is why all three are kept.

Keep all three passing when touching `adjoint_solver.py` or the objective.

## Shared fixtures

`tests/_helpers.py` (test-private, leading underscore) provides:

- `eval_obj(forward, T_obs, sigma, alpha)` — evaluates $J$ directly from the UFL forms,
  independently of `tao_solver`;
- `pick_random_test_direction(V, seed, scale)` — a random $\delta h$ in the function space;
- `h_true(x)` / `h0(x)` — the ground-truth and initial conductivities;
- `update_h(fwd, stepsize, delta_h)` — in-place `axpy` update plus `scatter_forward()`.

`tests/_tangent_solver.py` implements the forward-mode (tangent-linear) solve used by
`test_grad_forwarddiff.py`.

## Continuous integration

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/install-and-import.yml` | every push and PR | creates the conda env, `pip install -e .`, imports all modules |
| `.github/workflows/run-gradchecks.yml` | pushes/PRs touching `src/**`, `tests/**`, `pyproject.toml`, `environment.yml`; also `workflow_dispatch` | micromamba env (cached), `pip install -e ".[dev]"`, `pytest -q -m gradcheck --maxfail=1` |

Markers are registered under `[tool.pytest.ini_options]` in `pyproject.toml`.
