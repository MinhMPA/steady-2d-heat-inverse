# HMC Documentation Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain how the same physical inverse problem can be solved by optimization or posterior sampling, with reproducible examples and scientifically bounded uncertainty claims in README and Read the Docs.

**Architecture:** Retain the existing Sphinx/MyST documentation and flat scientific interfaces. Make `docs/hmc.md` the practical entry point, `docs/theory.md` the mathematical reference, and a short dated validation page the source of current evidence; preserve the longer validation reports as supporting history. Reuse the prepared conductivity figure and its compact data/provenance rather than rerunning inference.

**Tech Stack:** Sphinx, MyST, NumPy-style autodoc, Markdown, existing Jupyter example, Matplotlib SVG/PNG/PDF exports; the qualified DOLFINx/PETSc, JAX/BlackJAX environment for bounded example verification only.

**Spec:** [2026-11-16-hmc-documentation.md](../specs/2026-11-16-hmc-documentation.md), with the [review and evidence limits](../reviews/2026-11-16-hmc-review.md). Read both before execution.

## Global Constraints

- The current sampling qualification uses CPython 3.12, CPU float64, one MPI rank, BlackJAX 1.7.1, JAX/JAXlib 0.11.2 and ArviZ Stats 1.3.3.
- Preserve the native conda finite-element environment; inspect constrained optional-package resolution before installation.
- Preserve flat top-level imports, the existing legacy reconstruction semantics and the caller's supplied physical state.
- Use “Sampling” and “Optimization” in figure legends. Explain HMC in the README prose.
- Posterior uncertainty bands are equal-tailed pointwise 68% and 95% credible intervals of physical conductivity.
- Four-chain qualification requires R-hat < 1.01, bulk ESS >= 400, tail ESS >= 400 and no retained divergences for the monitored quantities; convergence alone does not establish calibration or continuum accuracy.
- Preserve immutable study inputs and saved chain histories. Documentation work must not launch new production studies.
- Do not edit the frozen calibration source tree. Implement documentation in an isolated working tree based on the reviewed implementation and any subsequently verified corrections.

## Review Focus

1. **An installation from master predates HMC:** code, linked notebook and advertised availability must refer to the same revision. Task 1 checks ancestry and page links; Task 6 checks rendered delivery.
2. **A previous result survives an interrupted extension:** readers must see the current incomplete status and historical results must be labelled. Task 4 requires the generation-aware regression result before documenting this as repaired.
3. **A 68/95% summary uses diagnostics computed for other quantiles or counts:** uncertainty must remain explicitly unqualified until a matching report exists. Task 2 executes the same probability tuple for both calls.
4. **A local build succeeds with mocked native imports, or MPI fails at exit:** runnable examples need bounded native execution and process-exit checks. Tasks 2 and 6 test these separately from Sphinx.
5. **The attractive mesh-32 figure is mistaken for finer-truth calibration or unbiased recovery:** provenance must disclose mesh-16 generating data and distinguish pointwise coverage from ensemble calibration. Tasks 1 and 5 verify the metadata and caption together.

---

## Starting point, file ownership and sequence

The reviewed code is `3df71cacf650f94d77c54fc0dbcf0f916825e199`. Remote master was `493e8dc09c4429a0a2c7cf265b3349dc5216ba61` on review; the user's main working tree is older `c8065ac` and contains prepared README/figure changes. Use the existing git-worktree skill at execution time to create an isolated working tree based on the actual implementation. Do not apply this plan directly to the old checkout or mutate the frozen calculation source. Copy only the intended prepared artifacts after inspecting the user's current diff; preserve `CLAUDE.md` and unrelated files.

| Files | Responsibility |
|---|---|
| `README.md`, `docs/index.md` | Shared scientific introduction and concise entry points |
| `docs/hmc.md`, new `docs/hmc-results.md` | Runnable inference sequence; saved-results and continuation guide |
| `docs/theory.md` | Legacy objective and new posterior, adjoint force and split dynamics |
| `docs/installation.md`, `docs/testing.md` | Environment, native execution/shutdown, actual validation commands |
| `docs/api/heat_inference.md`, `docs/api/index.md` | Public interface and result semantics |
| New `docs/validation/hmc-status.md`, existing three HMC validation reports | Dated evidence summary and supporting scientific history |
| `docs/_static/figures/conductivity-inference.{svg,png,pdf}`, `conductivity-slices.{csv,json}` | Prepared public figure and compact reproducibility data |
| `scripts/plot_inference_example.py` | Existing portable figure regeneration from exported values |
| `docs/conf.py`, `.readthedocs.yaml` | Existing documentation build settings; change only if needed for assets/import compatibility |

Tasks 1–6 run in order. Source fixes from the review are dependencies, not silently included implementation work. Each claim about a corrected behavior requires the relevant fix and its independent review; otherwise document the current limitation. A broad release-ready claim remains gated on all findings and required scientific evidence.

Commands below run from the isolated implementation working tree. `python` means the explicitly selected, recorded interpreter, not an arbitrary shell default. Put build/test outputs under a temporary directory. Use the clean compiler/transport settings from `docs/validation/hmc-mpi-shutdown.md` and the recorded qualification when testing on macOS; do not extrapolate them to other MPI installations.

### Task 1: Establish an honest, dated scientific evidence page

**Files:** Create `docs/validation/hmc-status.md`; modify the opening status summaries in `docs/validation/hmc-numerics.md`, `docs/validation/hmc-mesh-study.md`, `docs/validation/hmc-mpi-shutdown.md`.

**Interfaces:**
- Consumes: the pinned source revision, review findings, saved mesh-32 diagnostic record, calibration `progress.json`, figure provenance JSON, and actual test/build records.
- Produces: one dated status page with columns `Question`, `Evidence`, `Scope`, `Still required`, plus stable headings `Chain convergence`, `Repeated-experiment calibration`, `Mesh and prior studies`, `Numerical reproducibility`.

- [ ] **Step 1: Verify the target code and preserve the user's prepared work.** Record `git rev-parse HEAD`, `git status --short`, `git merge-base HEAD origin/master` and the list of reviewed corrective commits. Confirm `src/heat_inference.py`, `src/hmc_blackjax.py`, `src/posterior_samples.py` are present. Do not infer that HMC is on master from a local figure or branch name.
- [ ] **Step 2: Read authoritative records without modifying them.** Capture the current counts and diagnostic thresholds, source/numerical identities and exact study scope. At review the mesh-32 record had 4×1,024 draws, max R-hat 1.00312, min bulk/tail ESS 3615/2825, zero divergences; calibration had 8/100 qualified. Refresh these values at execution rather than copying them as live status. Use saved-state wording unless process liveness is checked.
- [ ] **Step 3: Write the evidence matrix.** Separate software checks, finite-solve accuracy, chain convergence, repeated-experiment coverage, mesh refinement and prior sensitivity. Mark incomplete/unverified outcomes explicitly. Include the compiler-option continuation limitation and other relevant review findings until their fixes pass. Preserve historical failed runs and the existing report details.
- [ ] **Step 4: Verify claim provenance.** Cross-check every numeric value against its source record and every “complete/passed/qualified” statement against the scope column. Confirm the figure's generating mesh is 16 and inference mesh 32. Export only small public summaries needed to substantiate claims; keep chain databases and machine-local paths out of reader dependencies.
- [ ] **Step 5: Review and commit the evidence page.** Expected: every claim has an evidence link or an explicit unavailable/incomplete label, and no inference campaign has been launched. Commit only the task's documentation as `docs: distinguish HMC convergence from calibration and mesh accuracy`.

### Task 2: Document and execute the complete physical workflow

**Files:** Modify `docs/hmc.md`, `docs/installation.md`; create `docs/hmc-results.md`. Verify against `tests/test_hmc_workflow.py`, `tests/test_hmc_results.py`, `tests/test_hmc_diagnostics.py`, `tests/test_hmc_mpi_shutdown.py`; these existing files are not changed solely to mirror prose.

**Interfaces:**
- Consumes: Task 1 status, `HeatInverseProblem(forward=..., observations=..., prior=..., numerics=...)`, `TemperatureObservations(points, values, sigma)`, `GaussianLogConductivity(mu, kappa, tau, h_ref)`.
- Produces: a coherent tutorial using `optimize(initial_h=..., maxiter=500, gtol=1e-6, ftol=1e-12) -> PosteriorEstimate`; `sample(initial=..., seed=..., chains=..., warmup=..., draws=..., output=..., controls=...) -> PosteriorSamples`; result/recovery guide using actual public methods.

- [ ] **Step 1: Assemble a self-contained small example.** Include creation of fine-data thermometers, declared prior, a separate inference forward model and context-managed inverse problem. Use explicit short demonstration settings and label the result unqualified; keep the production defaults in a distinct reference paragraph. Follow the existing nmesh-2 physical workflow test for bounded execution; do not silently use its coarse-problem step size as a universal recommendation.
- [ ] **Step 2: Make uncertainty probabilities and status agree.** Use `probabilities=(0.025, 0.16, 0.5, 0.84, 0.975)` in both `samples.diagnostics(quantiles=probabilities)` and `samples.summary(quantiles=probabilities)`. Show `estimate.success`, diagnostic status and summary qualification. Explain that the physical mean is not the exponentiated mean log field and that these are pointwise intervals, not a simultaneous 95% field envelope.
- [ ] **Step 3: Write the saved-result and continuation guide.** Cover `PosteriorSamples.open(output)`, `field(chain=..., draw=...)`, `load_draws(chains=..., start=..., stop=...)`, `sensor_predictions(chain=...)`, `temperature(chain=..., draw=...)`, `as_function(...)` and explicit `apply_to(forward, ...)`. Explain read-only array inspection versus optional diagnostic/FEM imports, `HeatInverseProblem.load(output)` and `resume(output, total_draws=N)` where N is the total retained count per chain. State unchanged physics/numerics requirements, partial histories, current diagnostics, rejection repeats and caller ownership.
- [ ] **Step 4: Correct installation guidance from actual constraints.** List the native conda prerequisite and separate sampling/diagnostics/plotting extras. Retain constrained dry-run inspection. Include the fresh-cache compile and bounded MPI-exit qualification; document the duplicate-library-path symptom and established clean compiler settings as platform-specific guidance. Do not install or upgrade dependencies during this documentation task without a separate need.
- [ ] **Step 5: Execute the documented short sequence in a disposable output directory.** Check that the returned fields have the correct node/coordinate association; the caller's h and T remain equal; a short one-chain example stays unqualified; matching requested probabilities are preserved; read/resume round trip succeeds; the process exits. Run `python -m pytest tests/test_hmc_workflow.py::test_documented_sequence_preserves_fields_and_saved_results tests/test_hmc_results.py tests/test_hmc_mpi_shutdown.py -q` under a declared external time bound. A timeout is a failed acceptance check, not a pass.
- [ ] **Step 6: Review and commit the tutorial.** Expected: a physicist can reproduce the sequence without missing variables and can distinguish numerical checks from scientific qualification. Commit as `docs: explain conductivity sampling and saved posterior workflows`.

### Task 3: Explain the posterior and its adjoint calculation

**Files:** Modify `docs/theory.md`, `docs/api/heat_inference.md`, `docs/api/index.md`; link from `docs/hmc.md` without duplicating derivations.

**Interfaces:**
- Consumes: actual definitions in `conductivity_prior.py`, `conductivity_posterior.py`, `hmc_blackjax.py`, and Task 2's public workflow.
- Produces: explicit equations for the sensor likelihood, whitening map, potential, adjoint gradient and split Hamiltonian, plus reference sections for the public input/state/result/control objects.

- [ ] **Step 1: Preserve and label the original reconstruction objective.** Add a separate new-posterior section; explain why its fixed-sensor likelihood and log-conductivity prior differ from the legacy integral misfit and conductivity roughness penalty. Do not claim the old TAO calls now optimize this posterior.
- [ ] **Step 2: Derive the new probability law and force in the notation of the code.** Use `m=log(h/h_ref)=mu+tau^-1 A^-1 M_L^1/2 z`, `A=kappa^2 M+K`, `U(z)=Phi(h(z))+z^T z/2`, and the sensor transpose without a likelihood mass matrix. State prior boundary conditions and parameter units. Explain the constant affine Jacobian between z and m and the distinct physical-h density.
- [ ] **Step 3: Explain the Hamiltonian calculation and responsibilities.** Describe exact Gaussian-prior rotations, likelihood kicks and the final Metropolis correction. Identify the BlackJAX momentum/acceptance/adaptation components actually called, the eager custom adjoint bridge, step-size-only warmup and duration interval `(pi/4, 3*pi/4)`. Explain failure handling and fixed solve tolerances. State that a trajectory contains many force evaluations.
- [ ] **Step 4: Complete the API reference with scientific semantics.** Check `HeatInverseProblem`, `InferredState`, `PosteriorEstimate`, `HeatNumerics`, `NodalField`, `TemperatureObservations`, `GaussianLogConductivity`, `PosteriorSamples`, `FieldSummary`, `DiagnosticReport`, `HMCControls`. Include array axes/coordinates, defaults, return meanings, optional imports, errors and closure. Keep private solver machinery out of introductory material. Explicitly document current defaults `chains=4`, `warmup=1000`, `draws=2000`, default initial spacing `6.5e-4`, bounds `(1e-6,1.3e-3)` and spread `1.0`, with their conditional qualification.
- [ ] **Step 5: Verify equations and interfaces against existing checks.** Run `python -m pytest tests/test_hmc_prior.py tests/test_hmc_posterior.py tests/test_hmc_optimization.py tests/test_hmc_jax_connection.py tests/test_hmc_dynamics.py -q` in the declared environment with a time bound. Compare public signatures directly with the rendered reference. Expected: tests complete successfully and the equations match the implemented discrete posterior; otherwise retain the unresolved limit.
- [ ] **Step 6: Review and commit the scientific reference.** Commit as `docs: explain shared posterior adjoint forces and split HMC`.

### Task 4: Make validation and recovery guidance match actual behavior

**Files:** Modify `docs/testing.md`, `docs/hmc-results.md`, `docs/validation/hmc-status.md`; inspect `notebooks/HMCSampling.ipynb` and `.github/workflows/hmc-cpu.yml`. Notebook/source corrections remain separately reviewed prerequisites.

**Interfaces:**
- Consumes: Tasks 1–3, actual current pytest selections/workflow, and corrected behavior or explicit limitations for the review's pilot, SQLite, notebook-generation, notebook-root and compiler-identity findings.
- Produces: a validation guide distinguishing analytic dynamics, adjoint tests, continuation/storage, native shutdown, notebook execution and long scientific campaigns.

- [ ] **Step 1: Replace obsolete test descriptions.** Remove the legacy “full 16-test suite” claim. Describe categories and exact commands instead of brittle totals. Correct the notebook description: the current test executes all computation cells with `plot=True` and checks image outputs; it does not simply skip plotting cells. State `hmc_study` selection and actual Linux CI verification status.
- [ ] **Step 2: Link each operational promise to its regression evidence.** Require a completed check of current versus historical extension reporting, root/notebooks working directories, SQLite disk-full admission, pilot memory exhaustion, effective compiler options, and closed chain memory before describing those issues as fixed. The source-fix review must provide exact test paths/results; if absent, retain a clear present limitation rather than inventing tests or results.
- [ ] **Step 3: Verify notebook paths and generation labels after their fixes.** Execute the final report cell from disposable root and notebooks directories with manufactured studies, including an unfinished extension. Assert the same study count, no report outside the selected project/study location, an `extension_in_progress` label, and historical labels on any old figures. Existing uncorrected behavior must fail these acceptance conditions.
- [ ] **Step 4: Verify the full notebook demonstration separately.** Run `python -m pytest tests/test_hmc_workflow.py::test_notebook_computation_cells_use_fine_data_and_coarse_inference -q` in a disposable checkout under its declared 600-second cell limit and an external process limit. Inspect both produced figures and require successful kernel/process shutdown. Do not execute against the frozen calibration checkout.
- [ ] **Step 5: Review and commit validation guidance.** Expected: no code test is presented as proof of unbiased recovery, and no stale report or timeout is presented as a successful current calculation. Commit as `docs: align sampling validation and recovery guidance with evidence`.

### Task 5: Integrate the minimalist conductivity figure and the two inference paths

**Files:** Modify `README.md`, `docs/index.md`; bring in the prepared `docs/_static/figures/conductivity-inference.{svg,png,pdf}`, `conductivity-slices.{csv,json}`, and `scripts/plot_inference_example.py`; adjust `docs/conf.py` only if needed to serve assets.

**Interfaces:**
- Consumes: existing exported slice CSV/JSON, already polished plotting script, Task 1 evidence, Tasks 2–3 explanations.
- Produces: one shared figure and consistent entry text on GitHub and Read the Docs, with a compact provenance note.

- [ ] **Step 1: Inspect and reuse the prepared artifacts.** Preserve the user's current README edits where scientifically correct. Verify CSV SHA256 against its JSON and confirm 4×1,024 draws, inference mesh 32, generating mesh 16, successful log-field mode optimization, physical sampling mean and equal-tailed probabilities. No new sampling or changed data selection.
- [ ] **Step 2: Write the same scientific introduction in both landing pages.** Explain conductivity recovery by optimization and sampling, HMC in prose and the shared discrete-adjoint gradients. Link the common-posterior tutorial and preserve a clearly labelled route to the legacy TAO example.
- [ ] **Step 3: Place the two central conductivity slices with minimal copy.** Keep “Sampling”, “Optimization” and “Truth”; one blue mean with darker 68% and lighter 95% bands, a distinct optimization line and quiet truth reference. Use friendly position/conductivity axes. Suggested caption: “Conductivity recovered by optimization and sampling. Shading shows pointwise 68% and 95% credible intervals.” Put mesh/provenance detail in a short expandable note on GitHub and a concise linked note in Sphinx.
- [ ] **Step 4: Verify reproducibility and visual meaning.** Run `python scripts/plot_inference_example.py` from the target checkout in the isolated plotting environment; verify data hashes remain unchanged. Inspect SVG and PNG at desktop and narrow widths, legend/band order, labels and truth alignment. Keep spectra out of the main figure until conjugate weighting, boundary/window convention and the distinction between a statistic of the posterior mean and a posterior distribution of that statistic are verified.
- [ ] **Step 5: Review and commit the landing pages/assets.** Expected: both methods and uncertainty are clear without dense captions; no claim of simultaneous coverage, finer-truth validation or calibration is implied. Commit as `docs: illustrate optimization and sampling with conductivity uncertainty`.

### Task 6: Build, inspect and hand off the complete documentation

**Files:** Modify `docs/index.md`, `docs/api/index.md` navigation as needed; inspect `.readthedocs.yaml`, `docs/conf.py`, `docs/requirements.txt`; record final evidence in `docs/validation/hmc-status.md`.

**Interfaces:**
- Consumes: all completed pages/assets and test records from Tasks 1–5.
- Produces: a warning-free, visually inspected local HTML build, reviewed documentation diff and an explicit delivery status.

- [ ] **Step 1: Complete navigation and revision-correct links.** Add `hmc-results` and `validation/hmc-status` to the toctree. Check every README/tutorial/notebook/source link resolves to a revision that actually contains HMC. Preserve existing useful anchors. Keep design plans excluded from publication.
- [ ] **Step 2: Run the strict local build.** Run `python -m sphinx -W --keep-going -b html docs /tmp/s2dhi-hmc-docs/html` and `git diff --check`. Expected: both exit zero with no warnings or whitespace errors. Also verify the docs-only environment described by `.readthedocs.yaml`; a full scientific environment can mask accidental runtime imports. Do not change the project's Python floor merely because the sampling qualification uses Python 3.12.
- [ ] **Step 3: Check links, assets and rendered interpretation.** Run `python -m sphinx -b linkcheck docs /tmp/s2dhi-hmc-docs/links`; inspect failures individually and distinguish inaccessible external sites from broken project links. Require every local link/image to resolve. Render README and HTML landing/tutorial/results/theory/API/status pages; inspect equations, mobile figure readability and the code/result distinction.
- [ ] **Step 4: Reconcile final evidence and review the whole diff.** Every pending repair or scientific campaign remains labelled. Verify only intended docs/assets/script changes are present, the original inference archives/source hashes are unchanged, and test timeouts are reported. Request an independent final scientific/documentation review with the exact fixed comparison point and the requirements document.
- [ ] **Step 5: Commit and report delivery precisely.** Commit only reviewed intended files as `docs: finish HMC documentation navigation and validation`. Report local build, example checks, remaining limits and whether master/Read the Docs has actually been updated. Do not merge, push, publish or start studies as an implicit part of writing documentation; follow the user's subsequent authorization for those actions.

## Self-review record

- Specification coverage: statistical definitions and gradients → Task 3; complete workflow/ownership/uncertainty → Task 2; current evidence and review limitations → Tasks 1/4; minimalist figure and provenance → Task 5; navigation, build, rendered review and delivery → Task 6.
- Interfaces checked against `3df71ca`: `optimize`, `sample`, `summary`, `diagnostics`, `load`, `resume`, `apply_to` and their named fields/parameters match the implementation. No new public numerical interface is proposed.
- Each of the five Review Focus conditions has an explicit verification in its owning task. Source defects have acceptance conditions, but their implementation is outside this documentation plan.
- Documentation checks use existing meaningful scientific tests and direct execution; no tests are added simply to match prose. No production inference is required.
- The preserved execution preference is task-by-task subagent implementation with independent review. This turn delivers the plan for user review; it does not execute it.
