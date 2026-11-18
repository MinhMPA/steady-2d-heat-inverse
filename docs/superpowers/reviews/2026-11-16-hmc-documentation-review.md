# Completed HMC documentation review, 16 November 2026

All six tasks in the approved documentation plan are complete on `docs/hmc-inference-guide`. The numerical source baseline is `3df71cacf650f94d77c54fc0dbcf0f916825e199`. Published documentation content was finally reviewed at `09c086c49882517bbf1c3f41876d8696073579b2`; the commit recording this report changes only this excluded review artifact.

Six independent task reviews approved their respective changes. The broad final review found no Critical or Important issue and four Minor issues. The single correction wave resolved all four; its independent scoped re-review found no new issue. Both full reports appear below. This approves the documentation for local handoff, not a scientific release, completed calibration or deployment.

## Delivery and preserved evidence

The branch and managed worktree are retained at `/Users/nguyenmn/.codex/worktrees/03e5/steady-2d-heat-inverse`. No merge, push, publication, production inference or edit to the frozen calibration source occurred. The original checkout's pre-existing edits were preserved. Numerical modules, tests, notebooks, constraints, environment.yml and pyproject.toml have no difference from the numerical baseline.

The committed slice CSV has SHA256 `75fbb1e756482f88520a5eb58b26aed604de26d7bed07e411e7cc1c4e082742a`, identical to its original export. The vector figure has SHA256 `da70da7f2a909cb62b00011cab94e0f0b74344d218aa792bf6d3efbeba6f746c`. The original export evaluates every physical draw on each central slice before taking quantiles. Neither values nor scientific archives were rewritten for this documentation work.

Fresh verification included 57 focused workflow/results/diagnostic/MPI tests, 168 prior/posterior/force tests, the exact tutorial and saved-result sequence, and the complete notebook. Counts are not assumed disjoint. The 168-test selection includes18 known PETSc option notices. The two known notebook-reporting defects were reproduced; their future acceptance fixture remains an expected nonzero result. The previous broad scientific-suite timeout remains unqualified evidence.

The final strict documentation-only Python3.11 build covered23 source pages without warnings. Local references, downloadable vector bytes and the corrected adjoint equation were checked. Earlier full local-link inspection covered7332 HTML targets and23 README targets; final changes received focused reference checks. Twelve external URLs resolved in the prior linkcheck with one disclosed ineffective command-line override; a corrected future recipe is recorded without rewriting that history. Desktop and narrow views were inspected, including measured scrolling and parameter wrapping. The temporary local MathJax preview does not establish hosted-CDN or accessibility behavior.

The successful final HTML build is a temporary local artifact at `/private/tmp/hmc-final-fix-html`. Public scientific evidence and limitations are retained in `docs/validation/hmc-status.md`. Plan-local scratch reports, checks and compiler caches are removed after this completed review, as the invoked workflow requires; receipt summaries below and the committed reports preserve their conclusions. Temporary preview images may remain under `/private/tmp` and are not enduring scientific inputs.

## Bounded-check receipts

Nonzero future notebook acceptance is the documented source limitation, not a repaired calculation. These are recorded execution outcomes, not freshly rerun tests during final review.

| Check | Exit | Seconds | External bound (s) |
| --- | ---: | ---: | ---: |
| `environment-smoke` | 0 | 1.870 | 60 |
| `task2-focused` | 0 | 482.833 | 600 |
| `task2-exact-docs-round2` | 0 | 42.927 | 600 |
| `task3-posterior-adjoint` | 0 | 9.613 | 600 |
| `task4-full-notebook` | 0 | 80.506 | 1200 |
| `task4-notebook-generation` | 0 | 0.293 | 180 |
| `task4-notebook-root` | 0 | 0.242 | 180 |
| `task4-notebook-future-acceptance` | 1 | 0.271 | 180 |
| `task6-linkcheck-final` | 0 | 3.837 | 180 |
| `task6-responsive-views-final` | 0 | 3.617 | 120 |
| `final-fix-sphinx` | 0 | 2.627 | 120 |
| `final-fix-local-view` | 0 | 0.912 | 120 |

## Controller decisions

- Ruling: The explicit approved plan authorizes a new managed worktree and read-only access to its named saved scientific inputs — the generic worktree consent and other-plan-artifact restrictions cannot prevent the plan's requested evidence refresh — if wrong, only reversible documentation/evidence copies need removal.

- Ruling: Existing source/notebook defects remain out of scope; documentation acceptance means truthful limitations with reproduced checks, not fixing the scientific code — spec explicitly permits documentation with pending fixes — cost is a later independent corrective change, visibly recorded.

- Ruling: Plan-specified commit subjects take precedence over the older CLAUDE no-prefix preference for these commits — user approved exact plan — cost is cosmetic history inconsistency.

- Ruling: Baseline is a strict Sphinx build plus existing same-revision 185-test evidence; only plan-named focused/native checks will be rerun — no numerical changes and broad suite previously timed out — cost is no new full-suite qualification, disclosed.

- Ruling: Use project-relative public links until the docs branch is published; verify local targets and disclose remote omissions rather than invent working remote URLs — cannot publish under this task — cost is remote checks remain explicitly pending.

- Ruling: Keep all SDD bookkeeping in the canonical parent-root workspace, using explicit plan/diff paths; implementation lives in the managed worktree — avoids two competing ledgers — cost is paths must be supplied explicitly to every worker.

- Ruling: Preserve exported CSV CRLF bytes and Matplotlib SVG path whitespace rather than normalize immutable artifacts merely for git diff --check — scientific data/provenance constraints outrank generic formatting — cost is generated-only whitespace notices, disclosed and excluded only from authored-text acceptance. Require committed CSV bytes to match provenance hash and authored files to pass diff --check. No .gitattributes change needed.

- Ruling: Task6 may minimally correct docs/usage.md along with README and tutorial links where inherited notebook targets are absent — final local-link acceptance and truthful availability require it — cost if wrong is reversible descriptive/link edit, no scientific change. Existing notebook references should use source-tree downloads for Sphinx. Absent EvaluateSolution.ipynb must not be advertised as present.

- Ruling: Finish by preserving the named documentation branch and worktree, without the generic integration menu or a repeated full scientific suite — the approved plan forbids implicit integration and requires focused documentation verification; all numerical sources are unchanged and earlier broad testing timed out — cost if wrong is a later explicit integration decision and no fresh full-suite qualification.

- Ruling: Accept all14 explicitly declined scientific/release judgments in final-review.md as outside this documentation delivery — they comprise the nine existing source defects, recomputation of frozen results, unfinished scientific campaigns, broader/platform qualification, hosted/accessibility validation and unrelated numerical re-audit — cost is those matters remain open; documentation approval is not scientific-release approval.

---

# Final scientific and documentation review

Reviewed 16 November 2026. Target: `/Users/nguyenmn/.codex/worktrees/03e5/steady-2d-heat-inverse`, branch `docs/hmc-inference-guide`. Fixed comparison: `3df71cacf650f94d77c54fc0dbcf0f916825e199..6153cdc844bf675310671b6c9fdbfc9c9cec3af0` (eight commits).

**Documentation readiness: APPROVED. Zero Critical, zero Important, four Minor findings.** The complete local documentation is ready for handoff. This verdict does not qualify the scientific implementation for release, finish the outstanding experiments, or authorize publication.

## Scope and method

Read the supplied final-review context, requirements, six-task plan, global constraints, reviewer instructions, ledger rulings and completed-task dispositions. Reviewed the changed scientific prose, tutorial, saved-result guide, reference, landing pages, installation/testing guidance, historical-record introductions, navigation, presentation rules, plotting script and provenance in separate passes. The newly included prior scientific review remains the authority for the nine pending numerical/operational defects.

Read the preserved Task 1, Task 5 and Task 6 reports and Task 5/6 independent reviews, final preservation record, and named native/build/link/rendering receipts. Inspected the committed figure and existing desktop theory and narrow README screenshots. No numerical calculation, build, link check, rendering run or scientific campaign was repeated. The implementation checkout, index, HEAD and scientific archives were not changed. This report is the only newly written artifact.

The one additional scientific doubt was whether the 129 positions on each mesh-32 slice represented quantiles of interpolated physical draws, or interpolation of nodal quantiles. Read the exact original export scripts supplied by the controller: `analysis/hmc-mesh32-hmc-vs-optimization-2026-11-16/prepare_readme_figure.py:37` evaluates each physical draw at the slice positions, then line 47 computes its quantiles; `plot_comparison.py:112` documents and implements the same order. The recorded export comparisons and preserved CSV hash bind those curves to the figure. This resolves the doubt without another calculation. No unchanged numerical module required inspection for an unresolved scientific or interface question.

## Strengths

- **One well-defined discrete posterior.** `docs/theory.md:131` separates the fixed physical thermometer likelihood from the legacy integrated misfit. The mass-lumped elliptic Gaussian prior, its boundary conditions and units are explicit at lines 156–192. Lines 194–213 correctly distinguish an affine change between whitened and log-conductivity coordinates from the nonlinear conductivity-density Jacobian. The mode in log conductivity and mean in physical conductivity are consistently distinguished in the tutorial and both landing pages.
- **The adjoint and Hamiltonian meanings are clear.** `docs/theory.md:215` gives the sensor transpose, constrained-node treatment, nodal derivative and transpose prior map without adding a likelihood mass matrix. Lines 256–259 correctly assign two heat-system solves to a gradient evaluation, not a trajectory. Lines 265–323 explain Gaussian rotations, likelihood kicks, the Metropolis correction, frozen retained controls, and the different treatment of nonfinite proposals and failed finite solves.
- **A researcher can follow the complete small experiment.** `docs/hmc.md:48` defines measurements, prior, numerical controls, optimization, sampling, matched quantile probabilities and caller-field preservation. The large coarse-example spacing and its one-chain uncertainty are explicitly unqualified. `docs/hmc-results.md:11`, `:65`, `:96` and `:115` distinguish reading saved physical arrays, private heat predictions, explicit application to the caller's model and continuation. Coordinate labels and the differences between E[h], exp(E[m]), T(E[h]) and E[T(h)] are preserved.
- **Evidence is separated by physical question.** `docs/validation/hmc-status.md:30` distinguishes discrete-solve accuracy, chain convergence, calibration, mesh refinement, prior sensitivity and numerical reproducibility. The saved 4 × 1,024 result is not pooled with older campaigns or presented as new diagnostic computation. The 8/100 calibration snapshot remains incomplete; “active” is not presented as a live-process observation. Mesh-16 generating data and mesh-32 inference are explicit at lines 54–62.
- **Known failures remain visible where they affect use.** `docs/hmc-results.md:148` warns that recorded identities omit effective compiler settings; lines 173–190 explain resource-exhaustion and notebook-generation limitations. `docs/testing.md:139` states exactly which future acceptance conditions still fail. Closing the enclosing inverse problem is distinguished from closing a saved-result handle. A successful notebook run is not used to claim that its final report selects the right study or generation.
- **The figure has scientifically appropriate content and provenance.** The committed CSV and original export share SHA256 `75fbb1e756482f88520a5eb58b26aed604de26d7bed07e411e7cc1c4e082742a`. The figure shows physical means, nested equal-tailed pointwise intervals, a separately styled log-field mode candidate and visible truth departures. It does not imply simultaneous field coverage or introduce unqualified spectra. The plotting script redraws the frozen export without heat solves or sampling.
- **Local verification is not confused with deployment or science.** The documentation-only Python 3.11 build is separated from the native Python 3.12 qualification. Same-source notebook downloads replace unsupported remote links. Source-preservation evidence shows unchanged numerical modules, tests, notebooks, constraints and environments. `docs/validation/hmc-status.md:177` reports the local construction and visual limits, including the temporary MathJax preview, native PETSc notices and unverified hosting/accessibility.

## Issues

### Critical — none

### Important — none

### Minor

1. **Inherited legacy adjoint multiplier convention has the wrong sign in its defining Lagrangian.** `docs/theory.md:59` writes L = J + <lambda,g> with g = -div(h grad T) - q. Stationarity then gives H lambda = -J_T, whereas the positive right-hand side at line 64 and negative conductivity-gradient term at line 85 use H lambda = +J_T. Change the defining expression to L = J - <lambda,g>, or reverse the definition of g. This is an inherited notation inconsistency, not an error introduced in the new fixed-sensor derivation or evidence of incorrect numerical gradients. Correcting it would help readers derive the legacy equation without an unexplained multiplier-sign change; no numerical rerun is needed.

2. **The README's inherited directory sketch contradicts the corrected notebook availability.** `README.md:159–160` still describes only the old test modules and lists `EvaluateSolution` among present notebooks, while line 147 correctly says that notebook is absent. Refresh the sketch to include the posterior modules/tests and `HMCSampling`, and omit the absent notebook or mark it historical there too. This is minor navigation confusion; the actual tutorial and downloadable links are correct.

3. **The two-slice figure is difficult to read quantitatively on a phone.** `README.md:38` and `docs/index.md:34` embed both panels at the available narrow width; the existing 390 px README screenshot confirms very small legends and tick labels. The caption and nearby prose keep the uncertainty interpretation clear, so this does not invalidate the figure. A direct “Open full-size figure” link to the vector image or PDF would give a reader a practical enlargement route. This retains the Task 6 minor observation.

4. **One optional link-check override is ineffective.** `docs/validation/hmc-status.md:192` correctly discloses the warning; `checks/task6-linkcheck-final/output.log:4` records the unsupported `linkcheck_rate_limit_timeout` override. Remove that override from the next routine invocation. All twelve checked external URLs resolved and the strict HTML build was warning-free, so this is a check-command cleanup, not a broken document or a reason to repeat the network check now. This retains the other Task 6 minor observation.

The existing 18 PETSc unused-option notices are not an additional documentation finding: their presence and limited significance are accurately recorded at `docs/validation/hmc-status.md:102` and `:216`. The documentation does not claim warning-free native execution.

## Verification assessment

The inspected receipts support the stated local result: the exact tutorial/results sequence returned zero in 42.93 s; 57 focused workflow/results/diagnostic/MPI tests passed; the posterior/adjoint selection returned zero with its recorded 168 passes and PETSc notices; the full notebook returned zero in 80.51 s; and the separate future notebook-reporting acceptance returned nonzero as documented. The selections are not assumed disjoint and are not added into a new full-suite total.

The final strict documentation receipt returned zero in 0.679 s after the responsive changes; the preceding complete-build receipt and environment record establish construction without native scientific dependencies. Task 6 records 7,332 resolved local HTML targets and 23 README source targets. The existing responsive evidence establishes real horizontal scrolling for wide equations/tables and contained document widths, rather than treating clipped screenshots as proof of usability. Desktop posterior, adjoint and Hamiltonian equations are visibly typeset in the inspected image.

The earlier broad scientific-suite timeout remains a timeout. The native checks do not repair the known defects, complete calibration, establish mesh convergence, or validate a new installation on another machine. Generated CSV line endings and SVG path whitespace are retained under the explicit provenance ruling; authored whitespace checks passed. The preservation record identifies clean target/frozen source states and no numerical-source delta.

## Recommendations

The four minor items can be addressed with narrow prose/link or future check-command changes. Preserve the current separation between posterior exploration, repeated-experiment calibration and discretization accuracy in any subsequent publication. No additional broad tests or inference campaigns are warranted by this review.

## Declined to judge

- Corrective implementation of omitted compiler-option identity: outside this documentation change; assessed the explicit continuation warning instead.
- Corrective implementation of pilot-memory admission: outside this change; assessed the instruction not to rely on continued admission after exhaustion.
- Corrective implementation of SQLite-storage admission: outside this change; assessed archive-preservation and recovery warnings.
- Corrective implementation of retained closed-chain fields/forms: outside this change; assessed the enclosing-problem closure guidance.
- Corrective implementation of the legacy requested initial temperature: outside this change and numerically unchanged; the dated evidence keeps the defect open.
- Corrective implementation of notebook extension-generation reporting: outside this change; the successful demonstration and failing future acceptance are explicitly separated.
- Corrective implementation of notebook root-directory selection: outside this change; the wrong-directory reproduction and warning remain visible.
- Corrective implementation of earlier optimization work undercounts: outside this change; the review/evidence does not claim corrected efficiency comparisons.
- Corrective implementation of earlier Fourier conjugate weights: outside this change; spectra were excluded from the new figure and the defect remains recorded.
- Fresh independent recomputation of archived diagnostics, optimization or slice quantiles: outside the authorized documentation verification and unnecessary for unchanged frozen exports; inspected the prior receipts, provenance and specific export-order source instead.
- Completed calibration, continuum accuracy, sharp-interface recovery and prior adequacy: require the separately declared unfinished scientific experiments; no such conclusion is asserted here.
- Full-suite success, three long Gaussian recovery cases, Linux CI and cross-platform sampling qualification: not established by these bounded checks; the missing evidence and previous timeout remain disclosed.
- Hosted Read the Docs content, actual GitHub rendering/badge reachability, CDN MathJax and accessibility: not established by local preview evidence; publication and those checks were outside this local delivery.
- Complete revalidation of unrelated inherited numerical modules: unchanged against the fixed baseline; no concrete new scientific doubt required that broader audit.

## Assessment

**Ready for documentation handoff: Yes.** All six documentation tasks meet their scientific intent and the local-delivery requirements, with the four nonblocking observations above. The guide truthfully describes the available posterior calculation and its limits; it must not be represented as a corrected scientific release or as completed calibration, mesh/prior validation, integration or publication.


---

# Scoped review after corrections

- **Inherited legacy adjoint multiplier convention has the wrong sign in its defining Lagrangian. — ADDRESSED.** `docs/theory.md:59–60` now defines `L = J - <lambda,g>` with the original residual `g = -div(h grad T) - q`. For the stated homogeneous adjoint boundary conditions, temperature stationarity gives `H lambda = J_T`, consistent with the positive forcing at `docs/theory.md:64–65` and negative conductivity contribution at `docs/theory.md:85–86`. This corrects the multiplier convention without changing the heat equation, likelihood, prior or numerical gradient. The existing focused image `/private/tmp/hmc-final-fix-visual/legacy-adjoint.png` visibly shows all three signs correctly.

- **The README's inherited directory sketch contradicts the corrected notebook availability. — ADDRESSED.** `README.md:161–173` now lists the fixed-thermometer heat calculation, log-conductivity prior, common posterior, sampling and optimization, saved draws and convergence summaries, together with the posterior checks and all five present notebooks including `HMCSampling`. `EvaluateSolution` is absent from this inventory and remains explicitly historical and unavailable at `README.md:149–150`. Read-only file enumeration confirms the newly listed modules, notebooks and `test_hmc_*` checks exist.

- **The two-slice figure is difficult to read quantitatively on a phone. — ADDRESSED.** `README.md:43` and `docs/index.md:41` provide the requested adjacent “Open full-size figure” links to the existing vector figure. The preserved local-check output confirms that the README target and generated landing-page download both resolve to SVG bytes with SHA256 `da70da7f2a909cb62b00011cab94e0f0b74344d218aa792bf6d3efbeba6f746c`. The correction leaves the figure and its scientific interpretation unchanged: physical-conductivity posterior mean, equal-tailed pointwise 68%/95% intervals, and a distinct log-conductivity mode candidate (`README.md:48–53`; `docs/index.md:43–47`).

- **One optional link-check override is ineffective. — ADDRESSED.** `docs/validation/hmc-status.md:197–203` supplies the next routine command using only `linkcheck_timeout=15` and `linkcheck_retries=1`; the unsupported `linkcheck_rate_limit_timeout` is absent from that recipe. The historical warning and twelve-URL result remain at `docs/validation/hmc-status.md:190–195`, and the original unsupported-override warning remains in `checks/task6-linkcheck-final/output.log:4`. No repeat network check is needed to resolve this finding.

## New Breakage in the Fix Diff

None. Count: **0** (Critical: 0; Important: 0; Minor: 0). The supplied `6153cdc844bf675310671b6c9fdbfc9c9cec3af0..09c086c49882517bbf1c3f41876d8696073579b2` correction contains only the four intended Markdown files. Its additions neither alter the physical posterior or archived calculations nor introduce calibration, continuum-accuracy, release or deployment claims.

## Checks Reviewed

- Read the supplied correction diff once, the four original findings, the correction report and the global constraints; inspected the affected source passages and the newly listed file names. No Git commands or numerical calculations were run for this re-review.
- Confirmed the preserved `checks/final-fix-sphinx/result.json` and `output.log` identify the target worktree, exact strict HTML command, exit 0, all 23 source pages and `build succeeded`, with no warnings. This verifies guide construction with mocked scientific dependencies; it is not additional posterior evidence.
- Confirmed the preserved `checks/final-fix-local-view/result.json` and `output.log` identify the covering script and exit 0, matching SVG hashes, passing local references/inventory checks, corrected rendered Lagrangian, retained historical warning and clean next-check recipe. Read that script to verify what its assertions cover.
- Read `/private/tmp/hmc-final-fix-visual/results.json` and inspected its existing equation image. The stored browser commands exited 0; the local-check output reports MathJax SVG without error nodes. No build, rendering, native test or network check was repeated.

## Out-of-Scope Observations

None.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** All four original findings are closed. This is the single scoped re-review of the final correction wave; no broader review was performed.
