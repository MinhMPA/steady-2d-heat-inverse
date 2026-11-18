# Testing

```bash
pytest -q -m "not hmc_study"  # ordinary validation; long scientific studies are separate
pytest -m gradcheck # adjoint-force checks only
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

## Input and physical-state checks

These small checks complement the heat/adjoint calculations. Plain `pytest -q`
also selects the other ordinary tests; it is not restricted to these modules.

| Test module | Covers |
|---|---|
| `tests/test_domain_coefficient.py` | DataFrame vs. `(N,3)` array parsing, case-insensitive column matching, missing-column errors, and `tab_interpolator` selection ("rbf" vs. "ct") |
| `tests/test_solver_validation.py` | constructor validation — the constant-`h` `TypeError`, `h_min=None` under `use_logh`, and the noiseless-plot error |

## Continuous integration

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/install-and-import.yml` | every push and PR | creates the conda env, `pip install -e .`, imports all modules |
| `.github/workflows/run-gradchecks.yml` | pushes/PRs touching `src/**`, `tests/**`, `pyproject.toml`, `environment.yml`; also `workflow_dispatch` | micromamba env (cached), `pip install -e ".[dev]"`, `pytest -q --maxfail=1 --durations=10` — the ordinary selection from `pyproject.toml`, excluding `hmc_study` and allowing missing optional-package skips; also builds documentation in a separate job |

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
`plot=True`. All code cells execute, including plotting and the final mesh-study
report cell; no plotting cells are skipped. The test checks the PNG signatures
of `conductivity.png` and `temperature-residuals.png`, retained array shape and
unqualified diagnostics. Synthetic observations
still come from the finer mesh at the same 256 interior sensors. This is an
execution check with explicitly unqualified diagnostics, not convergence evidence.

The isolated `.github/workflows/hmc-cpu.yml` job starts from `environment.yml`,
checks qualified native package versions, inspects constrained pip resolution
before installing optional extras, records actual versions and thread settings,
and requires zero skipped HMC tests. Source, constraints, notebook, tests, documentation and
workflow changes trigger it. Its actual selection is
`python -m pytest tests/test_hmc*.py -q -m "not hmc_study" --maxfail=1 --durations=10`;
it parses the resulting test report and fails if any selected test skips. The legacy job keeps optional sampling packages
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

## What each calculation check establishes

Use the qualified native environment and the [measured compiler and MPI
settings](installation.md#measured-macos-compiler-setup) for these commands.
The categories answer different physical questions; none establishes unbiased
conductivity recovery from noisy thermometers.

| Question | Exact focused selection | Limit |
| --- | --- | --- |
| Are the split dynamics correct on analytic targets? | `python -m pytest tests/test_hmc_dynamics.py -q` | Analytic targets, not heat-data calibration. |
| Is the discrete likelihood force consistent? | `python -m pytest tests/test_hmc_heat_model.py tests/test_hmc_jax_connection.py -q` and `python -m pytest -m gradcheck -q` | Discrete adjoint consistency; mesh and solver errors remain separate. |
| Are committed histories and continuation preserved? | `python -m pytest tests/test_hmc_store.py tests/test_hmc_store_creation.py tests/test_hmc_continuation.py tests/test_hmc_results.py -q` | Tested interruptions and archive operations, not every resource-exhaustion condition. |
| Do the measured native processes shut down? | `python -m pytest tests/test_hmc_mpi_shutdown.py tests/test_hmc_sbc_native_process.py -q` | Platform-specific native lifetime and transport checks. |
| Does the short illustrated experiment execute? | `python -m pytest tests/test_hmc_workflow.py::test_notebook_computation_cells_use_fine_data_and_coarse_inference -q` | One short, deliberately unqualified posterior; final report limitations below. |
| Do manufactured Gaussian studies recover the declared target? | `python -m pytest tests/test_hmc_gaussian.py -q` | Three long analytic recovery cases in the ordinary selection, not marked `hmc_study`; not rerun in this documentation update. |

The full notebook selection completed locally on 16 November in **80.51 s**,
with one test passing and successful kernel/process shutdown. It ran from a
disposable project copy with temporary results, a **600 s per-cell** limit and a
**1,200 s external process** limit. Both PNGs were visually inspected: physical
conductivity summaries, a retained-field temperature, and thermometer residuals
were visible with spatial labels. The existing illustrative conductivity image
uses a pointwise **90%** width; it is not the prepared 68%/95% figure described in
the tutorial. This documentation-only change does not alter that notebook.
The completed calculation is not evidence that its final report selects the
correct study directory or analysis generation.

## Open recovery and notebook limitations

The {download}`16 November review <superpowers/reviews/2026-11-16-hmc-review.md>`
reproduced six recovery and reporting defects. The subsequent
{download}`correction review <superpowers/reviews/2026-11-17-hmc-corrections.md>`
records their repairs and the related solver-option, work-count and Fourier
checks. The heading retains its earlier link target.

| Operational requirement | Corrected behavior and regression |
| --- | --- |
| Current versus historical extension reporting | The actual notebook cell consumes the shared generation-aware record and reports an unfinished extension with its current generation. Old figures are confined to identified history records. |
| Same studies from root and `notebooks` | Both working-directory checks find the manufactured study and write its report inside the detected project. |
| Stop after SQLite storage exhaustion | Real SQLite page-limit exhaustion stops further replica admission, including after restarting the coordinator. |
| Stop after pilot memory exhaustion | Memory, storage and SQLite resource failures stop the first spread attempt; ordinary numerical failures retain their separate treatment. |
| Match effective compiler settings | Effective DOLFINx/FFCx options are saved and bound to problem construction. Changed settings reject append while read-only archive access remains available. |
| Release closed-chain memory | Weak-reference checks show closed adjoint fields/forms are released even when the caller retains the closed chain handle. |

The earlier failing acceptance fixture remains historical evidence in the dated
review. Current regressions are in `tests/test_hmc_pr_review.py` and
`tests/test_hmc_sbc_execution.py`. They include fresh-process compiler choices,
private solver-option rejection, restoration of temporary PETSc options,
requested legacy starting temperatures, and odd/even Fourier-plane comparisons.
`tests/test_multifidelity_optimization.py` independently counts every trial solve,
including unsuccessful assembly attempts. Adaptive tolerances are compared by
measured accuracy and cost; a reduction in total work is not assumed.

After memory or storage exhaustion, investigate and restore capacity before
starting further calculations. Preserve committed histories and close owned
calculation contexts. Continuation now compares the process-wide PETSc options
database conservatively, so unrelated external option changes can also require
a new calculation. See [saved results and recovery](hmc-results.md).

## Fine-truth mesh and prior studies

`tests/test_hmc_validation.py` checks shared observation identity, the physical
interface-distance convention, predeclared one-at-a-time priors, reproducible
contiguous-block uncertainty, predictive observation noise, and visible missing
outcomes. Mock temperatures test bookkeeping only; they do not establish physical
truth refinement. The exact-continuation comparison excludes measured wall time
while comparing the scientific histories and solve counts exactly.

The long [mesh/interface calculations](validation/hmc-mesh-study.md) require both
truth records, all six baseline and eight sensitivity outcomes, actual-data force
qualification, ordinary four-chain diagnostics, report/notebook figure inspection,
and a warning-free Sphinx build. No absent or short calculation counts as Task 14
completion. Preserve failures and report insufficient independent blocks for cost
uncertainty as unresolved.

The Task 14 review regressions in `tests/test_hmc_mesh_review_fixes.py` check
unconverged evidence, abandoned physical work, interrupted result publication,
immutable truth-refinement records, explicit extension/reanalysis with preserved
history, and isolated figure rendering. The optional native test performs two
successive n4 quadrilateral short-chain physical summaries and figure renders,
checking native-library identity before and after rendering. Its deliberately
short chains remain unqualified and do not establish any Task 14 science outcome.
