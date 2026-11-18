# HMC and sampling review — 16 November 2026

The posterior definitions and adjoint connection agree with the accepted scientific design. Nine actionable defects were found in continuation, resource ownership, evidence presentation, compatibility and earlier comparison tools. Correct these before presenting the complete workflow as release-ready. None of these findings demonstrates bias in the saved mesh-32 posterior.

## Scope and method

- Reviewed implementation: `3df71cacf650f94d77c54fc0dbcf0f916825e199`.
- Fixed comparison point: remote `master`, `493e8dc09c4429a0a2c7cf265b3349dc5216ba61`, verified during this review. This is also the common ancestor; the complete comparison spans 80 files.
- The working implementation is `.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/nov11-calibration/source`. Its tracked contents remained clean. The main working directory remains on `feature/hmc-sampling` at `c8065ac`, before implementation.
- The comparison includes earlier multifidelity work. Findings about its benchmark and Fourier averages are identified below; they are not defects in the HMC transition itself.
- Three independent readers examined conformance to scientific specifications, project conventions/ownership, and concrete defects. Their two requested review axes remain separate below; additional findings are not substituted for those axes.
- Requirements: the 16 October HMC design, 17/18 October interviews, ADRs 0001–0003, the 26 October implementation plan, and subsequent numerical/shutdown evidence in the reviewed revision. Later qualified numerical choices supersede provisional step sizes in the original plan.
- No inference source, notebook, stored chain, dependency or compiler installation was modified. No changes were committed or published. Review experiments used manufactured data under `/private/tmp` and a disposable archive of the exact revision.

## Standards

**[P2] Count every linear solve in benchmark work totals — `src/benchmark_utils.py:125–126`.** The monitor records the most recent heat/adjoint iteration counts once per optimization iteration. Multiple line-search trials therefore disappear from the reported total. A real mesh-8 calculation with the default 50-iteration allowance reported 372 linear iterations while instrumenting every solve after setup counted 384; there were 32 heat/adjoint evaluations but 31 monitor entries. This compromises comparisons of computational efficiency. Count work where each solve occurs. This finding belongs to the earlier multifidelity changes in the merge comparison; short runs with one trial per iteration can agree accidentally.

**[P2] Stop admitting pilot calculations after resource exhaustion — `src/hmc_validation.py:1110–1125`.** The broad exception handler converts `MemoryError` into a failed initial-spread candidate and tries the next candidate. A controlled reproduction invoked both spreads 0.1 and 0.3 after the first exhausted memory. An exhausted machine is not evidence that one scientific control is unsuitable, and starting another calculation adds load. Propagate resource exhaustion to stop admission while preserving ordinary numerical failures as pilot evidence.

**[P2] Release completed chain workspaces from the physical problem — `src/heat_inference.py:234`.** Every independently created posterior is retained in `_workspaces` until the whole inverse problem closes. Closing a chain destroys its linear-solver objects but leaves its adjoint field and compiled gradient form reachable. Three created-and-closed chain workspaces, their fields and their forms all survived garbage collection; all were released only after `problem.close()`. Repeated sampling or continuation on one open problem therefore accumulates mesh-dependent memory. Deregister finished chain workspaces or fully release their owned field/form references.

No additional hard violation of the applicable project conventions was identified. These are concrete accounting and ownership findings, rather than a proposed architectural rewrite.

## Spec

**[P2] Preserve the chosen starting temperature in legacy solves — `src/forward_solver.py:427` (also line 224).** The design requires existing reconstruction calls to retain their numerical meaning. The new unconditional `setInitialGuessNonzero(False)` overrides both `petsc_opts={'ksp_initial_guess_nonzero': True}` at construction and a later direct setting before an ordinary solve. Both overrides were reproduced with the native mesh-2 solver. Users can no longer reuse their preceding temperature solution as requested. Confine compulsory zero starts to the private, deterministic posterior calculations.

The reviewed implementation matches the principal accepted definitions: every conductivity degree of freedom is inferred; fixed physical thermometers define a Gaussian likelihood; the Gaussian prior is on log conductivity; optimization and sampling share the potential in whitened coordinates; the supplied physical fields are preserved; and no extra conductivity-density Jacobian is inserted. Retained rejection repeats and explicit failed-solve behavior agree with the design.

No further concrete implementation omission was identified on this axis. Completed calibration, mesh refinement and prior sensitivity remain separate scientific evidence requirements. Their absence is not classified as an introduced programming defect. The earlier multifidelity changes are intentional preceding work, not HMC scope expansion.

## Additional independently reproduced defects

**[P2] Weight conjugate Fourier modes in the angular spectra — `notebooks/MultiFidelitySolve.ipynb:429–430`.** Equal weights on the stored `rfft2` half-plane do not represent equal weights on the full Fourier plane. With truth `cos(x)+cos(y)`, reconstruction `cos(x)` gives `T=r=0.57735`, whereas the rotated reconstruction `cos(y)` gives `0.81650`; the full-circle values are both `0.707107`. This was reproduced using the notebook's actual function. Include conjugate multiplicities or use the full transform. This concerns the earlier optimization comparison, and does not by itself establish an error in every later plotting script or diagnose the Hann-window question.

**[P2] Display the current study generation in the notebook — `notebooks/HMCSampling.ipynb:340–342`.** During an unfinished extension, the previous `result.json` remains present. The shared report correctly identifies `extension_in_progress`, but the subsequent notebook loop reads that old result directly and displays its “sampled” status, uncertainties and figures as current. Executing the actual cell with a manufactured extension reproduced the contradictory reports. Consume the generation-aware result or clearly label the previous result as historical.

**[P2] Stop admitting replicas after SQLite storage exhaustion — `src/hmc_sbc_worker.py:671–674`.** Exhausted SQLite storage raises `sqlite3.OperationalError` with `SQLITE_FULL`, without an OS `errno`. The resource classifier misses it and the coordinator admits another replica. A real SQLite page-limit failure passed through the actual coordinator and admitted both requested replicas after the first failure. Recognize SQLite storage exhaustion and halt new admissions while preserving completed evidence.

**[P3] Resolve the notebook study directory from its detected project root — `notebooks/HMCSampling.ipynb:327`.** Execution from the project root is supported, but the final cell searches `../runs`. A fixture containing one project study produced a zero-attempt report in the parent directory. Use the already detected project root and keep report output inside the intended study location.

## Parent verification finding

**[P2] Bind effective form-compiler settings to continuation identity — `src/posterior_store.py:92–121`.** Continuation records package/native identities and compiler environment variables, but omits options loaded by DOLFINx and FFCx from local or user JSON configuration. Two fresh processes with identical sources/environment produced identical `numerical_environment()` records despite effective C compilation flags changing from `['-O2']` to `['-O2', '-ffast-math']` and FFCx `epsilon` changing from `1e-14` to `1e-6`. These options enter generated finite-element forms, whose cached binaries are intentionally excluded from native identity. Record the effective options and compare them before appending, or explicitly freeze them. This demonstrates an undetected change in numerical settings; it does not demonstrate that an existing chain used differing settings or quantify a posterior shift.

## Validation and evidence boundaries

- Warning-free Sphinx build passed for all 21 source pages of the reviewed revision. Its mocked scientific imports do not establish that numerical examples execute.
- Native reproductions confirmed the legacy initial-guess override, retained chain fields/forms, benchmark work undercount and missing compiler-option identity. Each process exited successfully with the established macOS compiler settings and `FI_PROVIDER=tcp`.
- Independent small reproductions confirmed pilot memory admission, SQLite storage admission, both notebook reporting errors and the Fourier rotation discrepancy.
- The initial broad test attempt encountered duplicate `LC_RPATH` entries from inherited compiler flags. This environment failure was separated from source findings. A repeat using the recorded qualification settings, in a disposable source archive, reached its 1,200-second limit after 84 success markers without a completed test summary. It is not a passing full-suite run.
- Focused checks completed: **185 passed, zero failures/errors/skips, in 342.22 seconds**, with process return code zero. The selection covered `test_hmc_posterior.py`, `test_hmc_optimization.py`, `test_hmc_store.py`, `test_hmc_store_creation.py`, `test_hmc_dynamics.py`, `test_hmc_sbc_execution.py` and `test_hmc_mpi_shutdown.py`. All three explicit MPI-finalization subprocess cases passed. PETSc printed 18 unused-option notices at exit; these are retained in the log, not a basis for claiming warning-free native execution. The new review reproductions expose cases outside this existing test coverage.
- The three long manufactured-Gaussian chain-recovery tests and scientific `hmc_study` campaigns were not rerun. No new calibration or mesh-study campaign was launched. Linux CI and live Read the Docs publication were not verified.

The saved mesh-32 continuation reports four chains of 1,024 retained draws, maximum R-hat 1.0031196708971777, minimum bulk ESS 3615.030910154688, minimum tail ESS 2824.5840570818627, and zero retained divergences. This is a read of its saved status, not a fresh recalculation of diagnostics. The calibration progress file read on 16 November lists 8 qualified, 2 active and 90 not-started replicas; those are saved states, not a process-liveness check. Neither record establishes unbiased recovery for arbitrary conductivity fields or sharp interfaces.

## Reproduction records

Temporary parent evidence: `/private/tmp/hmc-review-2026-11-16/` contains `probe_native.py`, `native.out`, `probe_identity.py`, `identity-a.out`, `identity-b.out`, `run_probes.py`, compiler/test logs and the generated documentation. The parent probes used Python from `/Users/nguyenmn/miniconda3/envs/s2dhi-hmc/bin/python`.

Independent reproductions: `/private/tmp/hmc-review-spectrum.py`, `/private/tmp/hmc-review-notebook-generation.py`, `/private/tmp/hmc-review-notebook-root.py`, `/private/tmp/hmc-review-sqlite-worker.py`; pilot evidence is in `/private/tmp/hmc-review-resource-obhzbg0r/`. These paths are local review evidence, not public documentation dependencies.

## Documentation consequence

Use the companion [documentation requirements](../specs/2026-11-16-hmc-documentation.md) and [implementation plan](../plans/2026-11-16-hmc-documentation-update.md). Documentation can explain the available inference methods now, with explicit limits. Claims of reliable continuation, resource recovery, correct notebook reports or cost/spectral comparisons must wait for the corresponding corrections and their regression checks. No corrective implementation is included in this review-only task.
