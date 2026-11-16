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

The {download}`16 November review <superpowers/reviews/2026-11-16-hmc-review.md>` reproduced
all six issues below. No corrective source change or passing regression for
these specific findings is included in this documentation update. Existing
checks of neighbouring behavior must not be cited as their repair.

| Operational requirement | Current evidence and limitation | Required before claiming a correction |
| --- | --- | --- |
| Current versus historical extension reporting | Shared report identifies `extension_in_progress`; the final notebook cell still prints the previous generation as `sampled` and displays its old figures without historical labels. Reproduced again on 16 November. | Execute the actual final cell with an unfinished extension; require current status and explicit historical labels for every old figure. |
| Same studies from root and `notebooks` | A manufactured project with one study yields one from `notebooks`, but zero from the project root and writes a report outside that project. Reproduced again on 16 November. | Both working directories must find the same study count and write only inside the selected project/study location. |
| Stop after SQLite storage exhaustion | Review's real `SQLITE_FULL` reproduction admitted another replica. | Exercise SQLite page-limit exhaustion through the coordinator and assert no further admission. |
| Stop after pilot memory exhaustion | Review's controlled `MemoryError` reproduction tried the next initial spread. | Assert that memory exhaustion stops admission while ordinary numerical failures remain recorded. |
| Match effective compiler settings | Review found equal saved identities despite changed DOLFINx/FFCx settings. | Compare effective local/user options in fresh processes and reject a changed continuation identity. |
| Release closed-chain memory | Review retained adjoint fields/forms until the enclosing inverse problem closed. | Check field/form lifetime after chain closure and repeated continuation. |

A temporary acceptance check of both notebook working directories **failed as
expected** on the unchanged source: four required conditions failed (root study
count, report containment, current notebook extension label and historical
figure labels). Display was intercepted only to avoid requiring manufactured
PNG files; the actual final cell's study selection and evidence reads executed.
The shared report's pending label and the `notebooks` study count passed. These
are defect reproductions, not repaired-behavior regressions. Until the separate
source corrections are reviewed, avoid the final report cell on scientific
archives; inspect the explicitly selected study and its generation-aware report,
and retain previous figures as historical evidence.

After memory or storage exhaustion, investigate and restore capacity before
starting further calculations; do not rely on the coordinator to stop itself.
Preserve committed histories. Close the enclosing inverse problem between
repeated calculations, and independently preserve effective compiler settings
when continuing. See [saved results and recovery](hmc-results.md).

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
