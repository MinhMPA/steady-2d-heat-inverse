# Testing

```bash
pytest -q -m "not hmc_study"  # ordinary validation; long scientific studies are separate
pytest -m gradcheck # just the three slow gradient checks
```

## Gradient correctness

Gradient correctness is the thing worth testing most carefully here: an adjoint gradient
that is subtly wrong still *converges*, just to the wrong answer. It is therefore verified
three independent ways, all marked with the `gradcheck` marker.

The tests run real FEM solves on a $16\times16$ mesh and take minutes, not seconds.

### The three checks

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

### Shared fixtures

`tests/_helpers.py` (test-private, leading underscore) provides:

- `eval_obj(forward, T_obs, sigma, alpha)` — evaluates $J$ directly from the UFL forms,
  independently of `tao_solver`;
- `pick_random_test_direction(V, seed, scale)` — a random $\delta h$ in the function space;
- `h_true(x)` / `h0(x)` — the ground-truth and initial conductivities;
- `update_h(fwd, stepsize, delta_h)` — in-place `axpy` update plus `scatter_forward()`.

`tests/_tangent_solver.py` implements the forward-mode (tangent-linear) solve used by
`test_grad_forwarddiff.py`.

## Fast validation tests

Two modules carry no marker, so plain `pytest -q` runs them alongside the gradient checks —
they take seconds, not minutes.

| Test module | Covers |
|---|---|
| `tests/test_domain_coefficient.py` (8 tests) | DataFrame vs. `(N,3)` array parsing, case-insensitive column matching, missing-column errors, and `tab_interpolator` selection ("rbf" vs. "ct") |
| `tests/test_solver_validation.py` (5 tests) | constructor validation — the constant-`h` `TypeError`, `h_min=None` under `use_logh`, and the noiseless-plot error |

## Continuous integration

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/install-and-import.yml` | every push and PR | creates the conda env, `pip install -e .`, imports all modules |
| `.github/workflows/run-gradchecks.yml` | pushes/PRs touching `src/**`, `tests/**`, `pyproject.toml`, `environment.yml`; also `workflow_dispatch` | micromamba env (cached), `pip install -e ".[dev]"`, `pytest -q --maxfail=1 --durations=10` — the full 16-test suite, not just `gradcheck` |

Markers are registered under `[tool.pytest.ini_options]` in `pyproject.toml`.

## CPU conductivity posterior checks

In the separate constrained environment described in {doc}`installation`:

```bash
export S2DHI_HMC_QUALIFICATION=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
python -m pytest tests/test_hmc_workflow.py -q
python -m pytest -q -m "not hmc_study"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`test_hmc_workflow.py` first follows optimization → complete-state sampling →
physical summary → diagnostics → load/resume on a small real heat experiment.
Its `nbclient` check executes the notebook's computation cells using the same
interpreter and this checkout's source. There is exactly one `parameters` cell;
the check overrides `nmesh=4, chains=1, warmup=8, draws=12`, a temporary output and
`plot=False`. Only cells tagged `plotting` are skipped. Synthetic observations
still come from the finer mesh at the same 256 interior sensors. This is an
execution check with explicitly unqualified diagnostics, not convergence evidence.

The isolated `.github/workflows/hmc-cpu.yml` job starts from `environment.yml`,
checks qualified native package versions, inspects constrained pip resolution
before installing optional extras, records actual versions and thread settings,
and requires zero skipped HMC tests. Source, constraints, notebook, tests and
workflow changes trigger it. The legacy job keeps optional sampling packages
absent; sampling-dependent checks may skip there, while legacy and lazy-import
checks remain active. Full scientific studies carry `hmc_study`; the default pytest selection excludes
them, including in the existing legacy job. Run them explicitly with
`python -m pytest -m hmc_study` in a study environment. A local run does not establish that the Linux CI job has run.

Inspect actual physical figures separately. The notebook's Matplotlib figures use
recorded coordinates/connectivity and export small NumPy arrays. The existing
`test_plotting_function_keeps_linear_field_at_vtk_nodes` independently tests
physical DOF association for triangles and quadrilaterals at a non-dyadic mesh
resolution. This mapping check requires no installed PyVista/VTK renderer. A
separate optional plotting environment must be reported as unqualified until it
has itself been exercised; preserve the known-conflicting FEM environment.

Sampling convergence, fixed heat-solve accuracy, mesh approximation and prior
adequacy require separate evidence. The planned report is
`docs/validation/hmc-mesh-study.md`; no study results are asserted by these tests.
