# Conductivity posterior numerical qualification

The qualification apparatus passed two independent reviews. The first formal
mesh-16 force check failed before any trajectories were run. The three-mesh
numerical pilot and the 100 independent heat calibration experiments have
**not yet been completed**. Passing the helper tests would establish calibration
mechanics; it would not establish numerical accuracy or calibrated heat posteriors.

## The posterior being tested

The reference experiment uses quadrilateral finite elements on meshes 16, 32,
and 64, with respectively 289, 1089, and 4225 conductivity unknowns. Every
conductivity degree of freedom is inferred. Temperature is observed at the
same 256 locations `(i/17,j/17)`, for `i,j=1,...,16`, with independent known
Gaussian errors of standard deviation `1e-3 K`. The heat source is 1, the
bottom boundary temperature is 300 K, and the other walls are insulated.

The Gaussian prior has `mu=log(2)`, `kappa=2`, `tau=1`, and `h_ref=1` in the
declared conductivity units. These are recorded demonstration choices. The
potential is the sensor likelihood plus `z.T@z/2`; there is no mass matrix in
the likelihood and no extra conductivity Jacobian. The master seed is 20261026.

Before the finer-truth studies, the common numerical-pilot data are
`T=300+(y-0.5*y**2)/2`, for uniform conductivity 2 and source 1, plus Gaussian
noise drawn once with seed 20261024. `--data` accepts an existing `.npz` record
with `points`, `values`, and `sigma`, or a saved posterior directory whose
observation record has those same physical sensor positions and noise. The
record is copied once and then reused across all meshes.

Simulation-based calibration instead draws truths and noisy observations
from the same **discrete** prior and heat model used for inference on mesh 16.
This is the intended calibration experiment. Finer-truth/coarser-inference
comparisons assess a different question and belong to the subsequent studies.

## Temperature origin and the failed first force check

The first formal pilot preserved in `runs/hmc-pilot` failed all three candidate
solver tolerances on its mode candidate and ten prior-correlated perturbations.
At `(rtol,atol)=(1e-12,1e-14)`, only six of eleven fields passed: the mode's
normalized total-force error was `1.580094e-4`, its likelihood-force error was
`5.03162e-5`, and the largest energy discrepancy across the fields was
`2.37244e-4`. No trajectories were launched and no threshold was changed.

A diagnostic compared the same eleven fields and observations at temperature
origins 300 K and 0 K, including direct and mixed heat/adjoint solves. All 198
cases were evaluated. Subtracting 300 K from both the imposed boundary and the
data left the conductivity physics unchanged. At the shifted origin, all
fields passed at `rtol=1e-10`; at `rtol=1e-12` the maximum energy discrepancy
was `3.7136e-7` and the mode's force error was `9.423e-8`. Mixed solves isolated
the error to the heat solution: LU heat with iterative adjoint gave mode-force
errors about `9e-12`, whereas iterative heat with LU adjoint reproduced the
failure.

The physical right-hand-side norm changed from about 2652.79 to 0.0600566.
CG actually used a preconditioned convergence norm: the corresponding
preconditioned right-hand-side norms were about 4454.55 and 2.78995. Even LU
at the two origins differed in normalized mode force by `1.42479e-5`.
The matrices agreed exactly, but translating their assembled right-hand sides
left a norm defect of `3.91e-12`. Thus tighter CG stopping alone did not remove
the absolute-temperature assembly and solve roundoff. Evidence is in
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/task-13-offset-probe/probe.json`;
protected input hashes were unchanged.

The private posterior now solves for excess temperature
`theta = T - Tref`, where `Tref` is the first canonical imposed boundary value
(or zero if none is present). The reference is fixed independently of
conductivity, data and evaluation history. Nonconstant imposed boundaries
become `boundary_values - Tref`; the heat source is unchanged. The conductivity
force uses `grad(theta) = grad(T)`, and both likelihood and adjoint use the same
residual `B theta - (data - Tref)`. Physical fields and sensor predictions
return `Tref + theta` and `Tref + B theta`, with physical constrained field
values restored exactly. The frozen physical model, observations, recorded
fingerprints, prior, and public legacy heat solver retain their meanings.

Four portable regression cases (direct/iterative solves and constant/spatial
boundary values) failed temperature-origin invariance before this correction.
After correction, all 23 posterior checks passed in 2.99 s, including those
four cases, finite differences, Taylor remainders, an independent tangent
calculation, and failed-solve behavior. There were 18 known direct-LU
unused-option notices. The log is
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/task-13-temperature-green.log`.

A force-only diagnostic then evaluated the unchanged eleven saved fields and
original physical observations against LU. All fields passed at the two tighter
original tolerance pairs; the loosest pair still failed. Maximum discrepancies
across the eleven fields were:

| `(rtol,atol)` | `abs(delta U)` | Normalized total-force error | Normalized likelihood-force error |
| --- | --- | --- | --- |
| `(1e-10,1e-12)` | `1.8935185e-5` | `3.5538327e-6` | `1.1316755e-6` |
| `(1e-12,1e-14)` | `3.7136488e-7` | `9.4230949e-8` | `3.0006718e-8` |

The diagnostic exited successfully and all protected input hashes were
unchanged. Evidence is in
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/task-13-centered-force-probe/probe.json`
and the adjacent log. It uses the **pre-correction mode candidate without a
refit** and ten saved perturbations; it does not establish accuracy at a newly
fitted mode or at retained posterior states. The failed pilot is preserved, and
the changed numerical source requires a new identity and output directory.

The complete qualified suite then passed **456 tests and four subtests in
1363.75 s**, with zero skips or pytest warnings and 21 known direct-LU
unused-option notices. Its XML records 460 test cases with no failures, errors
or skips. Evidence is `task-13-temperature-full-suite.log` and `.xml` in the same
evidence directory. A strict clean build in the genuine docs-only environment
passed; Black on the three changed Python files and the patch whitespace check
were clean.

In the original numerical environment, the nine legacy modules plus posterior
checks passed **52 tests and four subtests in 4.17 s** in normal alphabetical
discovery order, with no skips or pytest warnings. An earlier explicit order
produced two multifidelity-optimization failures (50 other tests and four
subtests passed). The unchanged apparatus revision `165ffd5` reproduced those
same two failures in the identical order (46 other tests and four subtests
passed). This establishes a pre-existing order-dependent failure, not a
regression from the temperature correction. Native solver error-state
contamination after a deliberately failed solve is a supported explanation,
but that specific mechanism was not isolated experimentally. No legacy solver
code was changed. The preserved logs are `task-13-temperature-legacy.log`,
`task-13-temperature-baseline-legacy.log`, and
`task-13-temperature-legacy-ordered.log` (with XML for the ordered run).

Independent review of this correction, a fresh three-mesh numerical pilot and
all 100 calibration experiments remain required. Passing the regression suite
and saved-field diagnostic does not establish posterior qualification.

## Preparation measurements and provisional integration spacing

The controller measured the exact reference model with direct LU before
launching any trajectories. At the original optimization limit of 500, the
optimizer reported failure; its finite last state had `U=110.1948738`,
`Phi=105.2784960`, and `||grad U||=0.2128186`. This is a mode candidate.
The measured calculation used 589 heat and 589 adjoint solves in 0.2048 s.

With explicitly recorded `maxiter=5000`, `gtol=1e-6`, and `ftol=1e-12`, the
optimizer terminated successfully on the relative-objective criterion, at
`U=110.19475445`, `Phi=105.26422976`, and `||grad U||=0.00755923`.
The gradient criterion was not met. The measured calculation used 934 heat
and 934 adjoint solves in 0.3194 s.

Directional finite differences with spacings `1e-4` and `2e-4` gave largest
local Hessian eigenvalues 564202.8722 and 564202.9815. Their relative difference
was `1.94e-7`; the corresponding local harmonic spacing scale `2/sqrt(lambda)`
was about `0.002662639`. The reference controls therefore explicitly start at
`0.00065` with bounds `(0.000001,0.0013)`, approximately one quarter and one
half of that local scale. These are provisional controls, not a guarantee of
stability throughout the posterior or prior. The randomized trajectory-duration
law remains uniform on `(pi/4,3*pi/4)`.

These preparation measurements predate the final native-build identity and
are not formal numerical qualification. Evidence is held in
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/task-13-reference-probe`
and `task-13-reference-probe-5000`, including `probe.json`, `estimate.npz`,
and `observations.npz`; adjacent `.log` files contain progress. Root will add
immutable artifact checksums and final source/environment evidence after the
formal studies. No raw chains are committed to Git.

## Actual-build provenance and continuation

Each calculation records Python numerical-source hashes, package versions,
execution inputs, actual CC/CXX compiler version output and compiler executable
hashes. A fresh child of the active interpreter imports the available numerical
and finite-element components and enumerates the native images actually loaded.
Resolved image paths and SHA256 byte hashes identify the PETSc/MPI/HYPRE/BLAS/
DOLFINx/Basix build chain and loaded dependencies. Explicit SciPy sparse-linear
algebra and optimization imports also cover SuperLU and the mode finder. Available conda package records
supply version, build, archive SHA256, and record SHA256. Either supported MPI
provider is recorded as observed. Installed JAXlib native files are separately
labelled `installed_not_loaded` and hashed through distribution metadata; this
does not import JAX and does not claim runtime-backend inspection. Absence is
recorded explicitly.

Fresh-process resolution is distinguished from the parent's already-loaded
components. Corresponding loaded numerical components must agree with the fresh
resolution; a conflict stops continuation. The probe does not reuse a cached
image list or cached file hashes. Native LD_/DYLD_ loader controls remain recorded. Executable and source search
inputs enter numerical identity through the compiler, imports and libraries
actually resolved. Unused PATH/PYTHONPATH directories and the shell CONDA_PREFIX
label do not define numerical identity; the active interpreter and sys.prefix
identify the numerical installation. Project source contents, rather than the
project directory or Git revision, control source identity. Generated
problem-specific FFCx images and their cache-directory path are excluded: their
scientific identity is already tied to fixed forms, physical data, FFCx/compiler
and source evidence. Importing unrelated prior heat problems therefore does not
redefine the native shared-library identity.

Read-only inspection of saved chains still requires NumPy only. Sampling,
diagnostics, and plotting imports remain lazy. Changed physics, configuration,
numerical sources, native builds or declared controls require a new output.
The study manifest preserves the original experiment; ordinary restart resumes
its saved chain boundaries. Scientific failures remain visible and are not
silently rerun. A changed setting after failed force or dynamics qualification
must receive a new numerical identity and a new output directory.

The first native-evidence implementation passed five focused checks in 16.85 s
and nine continuation/persistence checks in 103.60 s, including exact fresh-Python
continuation. The completed native identity, including explicit SuperLU,
mode-finder and installed JAXlib evidence, then passed seven focused checks in
54.85 s. These included rejection of changed native/JAXlib bytes at unchanged
paths and exact fresh-Python continuation; there were no skips or warnings.

The complete study helper tests passed 21 checks in 31.67 s. An actual small
heat calculation exercised direct-LU comparisons, four short chains, saved
initial misfits/streams/statistics and an idempotent restart with unchanged
retained fields. It took 21.94 s, on mesh 2 with noise 0.05 K, four warmup and
six retained iterations per chain and explicitly separate toy integration
controls. Its diagnostics remained unqualified. Three known direct-LU notices
identified unused iterative-solver options; they were not failed heat solves.
This check does not supply reference-mesh timing or calibration evidence.

The controller logs are
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/task-13-native-focused.log`,
`task-13-native-continuation.log`, `task-13-native-amended.log`,
`task-13-validation-focused.log`, and `task-13-validation-amended.log`.
The apparatus-wide suite subsequently passed 449 tests and four subtests in
1314.52 s, with zero skips or pytest warnings. It emitted 21 known direct-LU
unused-option notices (18 from earlier checks and three from the toy study).
Its records are `task-13-apparatus-full-suite.log` and
`task-13-apparatus-full-suite.xml` in that same evidence directory. This full
suite precedes the search-location correction described below. Subsequent
independent apparatus reviews passed; formal scientific qualification remains
incomplete.

A subsequent continuation regression changed only an unused search directory.
Actual resolved numerical components remained identical, but the old identity
incorrectly rejected continuation because raw PATH had changed. The measured
regression failed as intended (one failure, six deselected, 8.99 s), recorded
in `task-13-search-red.log`. The corrected identity excludes unused executable/
source search entries and the shell CONDA_PREFIX label while retaining actual
resolution, native loader controls and changed-build/source rejection. The
post-correction focused set passed 21 checks in 136.36 s, with no skips or
warnings (`task-13-search-green.log`). Unused PATH/PYTHONPATH/CONDA_PREFIX changes
were accepted with identical resolved components; actual compiler inputs, form
compiler, source, native-library and JAXlib-artifact changes still rejected
continuation. Exact fresh-Python continuation also passed. The 449-test full
suite above is explicitly pre-correction; this focused set checks the amended
native-identity boundary.

Inspection cost was about 2 s per identity call in the initial native run;
the fresh-process continuation test took 21.67 s versus a historical 4.17 s.
This comparison shows overhead and is not a controlled performance benchmark.
Final qualification timing includes native provenance inspection. Both direct-LU
and candidate comparisons record actual elapsed time and heat/adjoint solve
counts; sampling-segment timing also records resumed and interrupted work.

## Numerical pilot

Run from the repository with the qualified native interpreter and environment:

```bash
python scripts/run_hmc_study.py pilot --config validation/hmc_reference.json --output runs/hmc-pilot
```

The mode candidate and ten seeded prior-correlated perturbations are evaluated
at each of `(rtol,atol)=(1e-8,1e-10),(1e-10,1e-12),(1e-12,1e-14)`, against direct
LU on identical fields. Acceptance requires `abs(delta U)<1e-4`,
`norm(delta grad_U)/max(1,norm(grad_U_ref))<1e-5`, and separately the same
relative criterion for `grad_phi`. The loosest passing pair is provisional
until it also passes comparisons on ten actual retained posterior states.

At mesh 16, four chains are run for each of the spreads 0.1, 0.3, and 1.0,
with 256 warmup and 256 retained iterations per chain. The largest qualifying
spread is selected. Every chain must have finite starts, no failed PDE, zero
divergences in the final 64 warmup and all retained iterations, no integration
spacing at its lower bound, and mean retained acceptance at least 0.65.
A failed spread trial is preserved, and the other predeclared trials remain
part of the experiment. If none qualifies, the study stops for investigation.

Meshes 32 and 64 then use the selected spread with the same four-chain pilot
lengths. Their direct-LU comparisons also include ten actual retained states.
These are numerical pilots, not full production convergence studies.
`qualification.json` records fixed solver choices, all comparisons, initial
misfits, acceptance, energy errors, spacings, actual trajectory durations,
solve counts, timings and immutable chain checksums. `events.jsonl` exposes
committed iteration counts and progress during the calculation.

Production begins with 1000 warmup and at least 2000 retained iterations per
chain. The initial estimate is `max(2000,2*ceil(256*400/min_pilot_ESS))`;
observed pilot timing supplies a labelled cost extrapolation. Subsequent
finite, insufficient ESS drives extension of the same chains using
`2*ceil(current_draws*400/min_ESS)`, with at least a doubling when another
ordinary convergence condition still fails. Retained divergences, failed
solves, nonfinite diagnostics and a lower-bound spacing require investigation.
Appending draws cannot erase them.

## Same-discrete-model calibration

```bash
python scripts/run_hmc_study.py sbc --config validation/hmc_reference.json --output runs/hmc-sbc --replicates 100
```

`pilot_output` in the configuration must identify the completed matching pilot;
set it before creating the experiment if a different output location is desired.
The present apparatus runs the predeclared replications sequentially. Any future
parallel execution must preserve explicit replicate indices/seeds and receive
a new study identity and review; repeated one-replicate commands would repeat
replicate zero and would not supply independent calibration experiments.
Each replicate records independent truth, noise, sampling and tie-breaking
seeds. Optimization always starts from the prior median field, with termination
status retained. Four-chain qualification requires rank-normalized split/folded
R-hat below 1.01, combined bulk and tail ESS at least 400 for every monitored
nonconstant physical quantity, finite Monte Carlo errors and zero retained
divergences. Difficult prior draws are preserved even when they fail.

The rank quantities are area-weighted mean conductivity and conductivity at
`(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)` and `(0.75,0.75)`. Following
[Talts et al., Section 5.1 and Algorithm 2](https://arxiv.org/html/1804.06788v2),
the study estimates ESS for indicators at 19 empirical quantiles from 0.05
through 0.95. If an indicator's ESS exceeds the raw draw count, indicating
possible antithetic sampling, it first subsamples by two and recomputes ESS.
One interval comes from the worst indicator ESS across all five quantities.
Twenty-five values per chain give 100 rank draws. The same complete original
chains must also provide 100 values at twice that interval; otherwise they
are extended and requalified. Every original draw, selected index, residual
physical/indicator autocorrelation and doubled-interval rank is preserved.
Exact ties are randomized uniformly over all admissible rank positions.

The formal CDF diagnostic simulates 100000 discrete-uniform rank ensembles,
with seed 20261027 and ranks 0 through 100. A maximum absolute CDF deviation
produces a simultaneous envelope across thresholds; Bonferroni allocation of
0.05 across the five reported quantities gives nominal familywise 95% coverage.
The doubled-interval calculation is a sensitivity diagnostic. Its discrepancy
also triggers investigation; it is not treated as another independently
allocated formal test. Rank histograms and CDF-deviation figures are saved as
PDF and PNG. Envelope violations are preserved and investigated rather than
rerun until they pass.

One hundred replications have limited sensitivity. Absence of a detected
calibration discrepancy is conditional evidence, not proof of exact sampling.
Missing, failed or unqualified replications prevent a claim of complete
calibration. Small replicate counts exercise mechanics only.

## Four separate scientific conclusions

| Question | Current status |
| --- | --- |
| Numerical correctness of fixed discrete posterior evaluations | Formal three-mesh comparisons pending |
| Sampling convergence | Actual production diagnostics pending |
| Mesh approximation to the physical solution | Not assessed here; subsequent finer-truth studies |
| Sharp-truth prior adequacy | Not assessed here; subsequent modelling studies |

The required 100-replication calibration remains incomplete. This report will
be updated with actual source/environment records, commands, figures,
measured costs, failures, outcomes and immutable manifest/chain checksums
after the controller executes and reviews the scientific studies.
