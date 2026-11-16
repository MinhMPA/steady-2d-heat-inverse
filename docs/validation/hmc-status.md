# Conductivity inference: evidence as of 16 November 2026

Noisy temperature measurements constrain conductivity through the steady heat
equation. Agreement among posterior chains tests how well we have sampled one
chosen discrete posterior. Repeated synthetic experiments test whether its
uncertainty statements recover their declared probabilities. Refining the mesh
and changing the physical prior test further, distinct approximations.

The evidence below was read on **16 November 2026**. The sampling source is
`3df71cacf650f94d77c54fc0dbcf0f916825e199`; the documentation starts from
`aeee1b22777fe7b2241b945fde68b39728b357fd`. The common ancestor with the inspected
`origin/master` is `493e8dc09c4429a0a2c7cf265b3349dc5216ba61`. This identifies the
reviewed branch evidence; it does not establish that sampling is on `master`.
No corrective implementation commit follows the reviewed source in this
snapshot. The limitations in the {download}`16 November review
<../superpowers/reviews/2026-11-16-hmc-review.md>` therefore remain open.

```{raw} html
<style>
#evidence-matrix table {table-layout: fixed; width: 100%;}
#evidence-matrix table th, #evidence-matrix table td,
#compact-saved-record-provenance table th, #compact-saved-record-provenance table td {
  white-space: normal;
  overflow-wrap: anywhere;
}
#compact-saved-record-provenance table code {white-space: normal;}
</style>
```

## Evidence matrix

| Question | Evidence | Scope | Still required |
| --- | --- | --- | --- |
| Do the numerical and inference checks pass? | The [11 November checks](hmc-mpi-shutdown.md#verification-record-and-limits) completed with 630 tests and 32 subtests in the HMC environment; 456 tests and 32 subtests passed with 29 skips in the original environment. The 16 November review completed 185 focused tests. | Different dated selections and source states; these counts are not additive. The review's broad repeat timed out after 84 success markers without a completed summary. | Regression checks for the newly reproduced review defects and a completed full-suite check of their corrections. |
| Is the finite heat/adjoint solve sufficiently accurate? | [Numerical force comparisons](hmc-numerics.md#temperature-origin-and-the-failed-first-force-check) passed declared tolerances on selected fields; matching pilots covered meshes 16, 32 and 64. | Accuracy relative to direct LU on the same discrete problem and the tested states. Historical failed force checks remain recorded. | Actual-data force qualification for each new physical experiment; a continuum error assessment remains separate. |
| Have the mesh-32 chains converged by the declared diagnostics? | The [saved diagnostic snapshot](#chain-convergence) records four chains of 1,024 retained draws, maximum R-hat 1.0031196708971777, minimum bulk/tail ESS 3615.030910154688 / 2824.5840570818627, and zero retained divergences. | Saved qualification of monitored quantities for one discrete posterior; diagnostics were read, not recomputed. | Independent calibration, mesh refinement and prior assessment; continued attention to Monte Carlo errors for the quantities being interpreted. |
| Do posterior intervals have their declared repeated-experiment probabilities? | The [saved calibration snapshot](#repeated-experiment-calibration) has 8 qualified, 2 active and 90 not-started replicas out of 100; its status is incomplete. | Same-discrete-model simulation-based calibration on mesh 16. “Active” is a saved state; process liveness was not checked. | Complete and assess the declared 100-replica experiment; no calibration conclusion is supported yet. |
| Are conductivity inferences stable under mesh refinement and prior changes? | [Frozen smooth and sharp reference observations](hmc-mesh-study.md#frozen-reference-observations-2026-11-07) meet the declared sensor-refinement criterion. | Qualification of reference temperatures; the six baseline posteriors and eight prior-sensitivity outcomes remain incomplete in the available evidence. | Data-specific numerical qualification and all fourteen scientific outcomes, with differences compared against their separate Monte Carlo errors. |
| Can the calculation be reproduced and continued without a changed numerical experiment? | [Shutdown and fixed-field comparisons](hmc-mpi-shutdown.md) support the measured TCP setup; [saved numerical identities](hmc-numerics.md#actual-build-provenance-and-continuation) reject many changed inputs. | Machine- and version-specific measurements. The review reproduced an omitted effective form-compiler configuration. | Correct and verify compiler-option identity before treating continuation checks as complete; resolve the other review limitations below. |

## Chain convergence

The saved mesh-32 status reports `state="converged"`, `qualified=true`,
`returncode=0`, and retained counts `[1024, 1024, 1024, 1024]`. The maxima and
minima in the matrix are copied from that status record. No new trajectories
or diagnostic recomputation were performed for this page.

The four-chain qualification requires **R-hat < 1.01**, **bulk ESS >= 400**,
**tail ESS >= 400**, and **no retained divergences** for the monitored
quantities. These conditions bound evidence about finite-chain exploration of
the selected posterior; they do not establish repeated-experiment calibration,
continuum accuracy, or recovery of arbitrary sharp conductivity interfaces.

The prepared central horizontal and vertical conductivity slices use this
**32 by 32 inference mesh** and a **16 by 16 generating mesh**. They are a
separate demonstration from the finer-truth smooth/sharp mesh study. Their
sampling curve is the mean of physical conductivity, with equal-tailed
pointwise 68% and 95% credible intervals (16/84 and 2.5/97.5 percentiles).
Those intervals concern each location separately and are not simultaneous
bands over the whole field. The optimization curve is a successfully
terminated mode candidate in log conductivity. Convergence of an optimizer
and chain convergence answer different questions.

## Repeated-experiment calibration

The current saved calibration record requests 100 replicas and contains 100
entries: **8 qualified, 2 active, 90 not started**, with status `incomplete`.
Each replica draws a conductivity truth and noisy thermometers from the same
discrete mesh-16 prior and heat model used for inference. Qualification of an
individual replica means its posterior diagnostics passed; calibration requires
assessing the ensemble of posterior ranks or inclusion probabilities.

The older [numerical record](hmc-numerics.md#matching-numerical-pilot-and-calibration-progress-2026-11-05)
preserves a different, interrupted campaign with ten qualified posteriors,
a failed warmup and cooperative stops. Those historical counts must not be
pooled with the present eight or substituted for the current snapshot.

## Mesh and prior studies

The [mesh-study record](hmc-mesh-study.md) defines six baseline outcomes
(smooth/sharp truths at inference meshes 16, 32 and 64) and eight sensitivity
outcomes at mesh 16. Their production and prior-sensitivity conclusions remain
**incomplete** in the available evidence. Qualified reference observations
alone do not qualify a posterior calculation.

The smooth reference uses mesh 256; the sharp reference uses mesh 1024. Their
last sensor-temperature refinement differences are respectively
`0.00838298677763305*sigma` and `0.0774712140128031*sigma`, below the declared
`0.1*sigma` criterion. This controls a measured temperature difference at the
sensors; it is not a rigorous continuum bound on the recovered conductivity.
A Gaussian log-conductivity prior and continuous nodal fields can smooth a
material interface even with well-converged chains. Mesh effects, prior
sensitivity, Monte Carlo uncertainty and pointwise posterior spread must be
reported separately.

## Numerical reproducibility

The current qualification uses **CPython 3.12, CPU float64, one MPI rank,
BlackJAX 1.7.1, JAX/JAXlib 0.11.2 and ArviZ Stats 1.3.3**. The
[shutdown record](hmc-mpi-shutdown.md) gives the measured macOS transport and
native-library conditions, successful process exits and limits. Passing
numerical tests does not establish calibration. Native test logs contain
known PETSc unused-option notices; the 11 November HMC suite also reported
three XML-metadata warnings. The 185 focused checks from 16 November had no
failures, errors or skips, but emitted 18 PETSc unused-option notices. Native
execution is therefore not described as warning-free.

The 16 November review reproduced equal recorded continuation identities
when effective local/user DOLFINx and FFCx settings changed: compilation flags
changed from `['-O2']` to `['-O2', '-ffast-math']`, and FFCx `epsilon` from
`1e-14` to `1e-6`. These settings affect the generated finite-element forms.
Until their identity is corrected and verified, a matching saved identity
alone cannot certify unchanged numerical settings. This finding demonstrates
an undetected change; it does not establish a bias in the archived mesh-32
posterior.

Other open review findings concern continued admission after pilot memory
exhaustion or SQLite storage exhaustion, retained fields after chain closure,
the legacy solver's requested initial temperature, and notebook reports using
an earlier analysis generation or the wrong study directory. Earlier
optimization comparisons also undercount line-search solves and misweight
conjugate Fourier modes. The dated review describes the reproductions; none
of these issues is silently treated as fixed here. Historical failed
calculations and shutdown errors remain in the linked records.

The review's warning-free Sphinx build covered 21 source pages with mocked
scientific imports. It verifies documentation construction, not native
execution of examples. Linux CI, live Read the Docs publication, the three
long manufactured-Gaussian recovery tests and new scientific campaigns were
not verified in that review. This documentation update launches no campaign.

## Documentation-update notebook checks

On 16 November the exact notebook computation test completed in **80.51 s**
with one test passing in a disposable source copy. It used the qualified native
interpreter, 600 s per-cell and 1,200 s external limits, executed all code cells
with `plot=True`, produced both PNGs, and completed kernel/process shutdown.
Both images were inspected. The short one-chain posterior remains unqualified;
the notebook's illustrative conductivity spread is a pointwise 90% width,
separate from the prepared 68%/95% conductivity slices above.

Separate manufactured studies exercised the actual final notebook cell from
both supported working directories. The `notebooks` case found one study and
its shared report marked an unfinished extension, but notebook output still
presented old `sampled` evidence without historical figure labels. The root
case found zero studies despite one project manifest, writing into its parent.
The combined future-acceptance assertion failed as expected. The successful
full notebook execution therefore does not establish correct report location
or current-generation evidence. The [validation guide](../testing.md#open-recovery-and-notebook-limitations)
lists the precise conditions required after separately reviewed source fixes.
No new long scientific study or Linux CI result is asserted.

## Compact saved-record provenance

The scalar summaries above were refreshed directly from the saved records on
16 November 2026 and matched the compact evidence capture at
`2026-11-16T14:21:50.653281+00:00`. The following SHA256 values bind the read
records. They allow later custodians to identify the evidence without making
machine-local directories or chain databases prerequisites for this page.

| Saved record | SHA256 |
| --- | --- |
| Calibration `progress.json` | `8940950f4490d7fde6168e6ff22d1e01302aae2ecb3eed17fb90f3d891d02a42` |
| Mesh-32 `status.json` | `44f5bc12f0cae5dea327a3e4451b81276c50ff7fb172447553dd40cc95abd9b7` |
| Conductivity-slice provenance JSON | `0404040e03a10122acb438507238ea045c4636d35e08f24df8f724320c01f9d0` |
| Conductivity-slice exported CSV, as declared in its provenance | `75fbb1e756482f88520a5eb58b26aed604de26d7bed07e411e7cc1c4e082742a` |

The slice provenance identifies physical problem
`58b44bcccf6a654c7847c7ba9b7c443c30c50e2d4b211d535ecfb20f84e8dc41`
and numerical calculation
`a83eee0965795535382df72ba21d5de296d44e1755607d66a2c30cb3e75c21ea`.
The source records remain immutable. Their small published summaries support
the dated claims here; they do not replace the full histories for an
independent recalculation.


## Local documentation verification and delivery

The documentation update on `docs/hmc-inference-guide` was constructed on
16 November 2026 with Python 3.11.15, Sphinx 9.0.4, MyST Parser 5.2.0,
NumPy 2.4.6 and the Read the Docs theme 3.1.0. The strict HTML build treated
warnings as errors and completed without warnings. DOLFINx, PETSc, MPI, JAX,
BlackJAX and SciPy were absent from this documentation environment; mocked
imports therefore did not supply native scientific validation. The project's
Python floor remains 3.11; the separate sampling qualification uses Python 3.12.

All local HTML links, assets and fragments were checked, together with README
source links. Existing notebooks are downloadable from this documentation's
source revision. The historically advertised `EvaluateSolution.ipynb` is absent
from this checkout and is identified as unavailable. An external-link check
resolved all twelve remaining documentation URLs, including one PETSc redirect.
Its optional command-line rate-limit override was unsupported and produced one
configuration warning; no broken external link was suppressed. An earlier check
of superseded notebook URLs encountered missing GitHub targets and rate limiting;
those links were replaced with downloads from the available source tree.

Desktop rendered openings were inspected for the README, landing page, posterior
tutorial, saved fields, theory, public reference and this evidence page. Equation previews
use a disposable copy with a local MathJax 4 SVG bundle and its calligraphic font,
because the local headless browser could not complete the CDN TLS handshake.
Published MathJax configuration is unchanged. Optional speech-worker requests
failed in that preview, so this is visual typesetting evidence rather than an
accessibility or hosted-CDN check. The tutorial likelihood/prior and opening
heat/legacy equations were visibly typeset. Complete narrow-page inspection and
deeper posterior/force/Hamiltonian equation inspection remain unverified after
the rendering tool stalled at approval. The earlier figure-only narrow preview
retained both conductivity slices; long equations and calculation listings may
require horizontal scrolling. The public-reference tables were changed to wrap
after the desktop preview exposed clipped descriptions; their final visual
confirmation also remains pending.

The native evidence completed earlier in this update is separate: 57 workflow,
saved-result, diagnostic and MPI checks; 168 prior/posterior/force checks;
the exact short tutorial and saved-result sequence in 42.93 s; and the complete
notebook test in 80.51 s. The 168-check selection emitted 18 known PETSc option
notices. The notebook reporting acceptance conditions still fail as described
above. None of these checks completes calibration, mesh refinement, prior
sensitivity or the outstanding source corrections. No production studies were
launched, and saved scientific records were preserved.

This is a local documentation result. No merge, push, update of `master`,
Read the Docs deployment or hosted HMC availability is asserted.
