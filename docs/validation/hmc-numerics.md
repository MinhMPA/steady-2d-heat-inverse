# Conductivity posterior numerical qualification

The first formal mesh-16 force check failed before any trajectories were run.
After the temperature-origin correction and independent review, the numerical
pilot at source `40f1324` completed on meshes 16, 32 and 64. Its evidence is
summarized below. The matching pilot for the revised integration, defaults and
calibration execution completed at source `7196c0a` on all three meshes. The 100 independent
heat calibration experiments remain **incomplete**. The campaign is preserved
after ten qualified posteriors, one failed warmup and two cooperative interruptions.
Fixed-start observations support inadequate Hamiltonian time resolution as the
explanation of the failed warmup. Both declared smaller-spacing pilots are
running from source `4853edc`; neither is complete or qualified. That source
includes a reviewed correction that preserves the numerical experiment through
figure production and rejects inconsistent numerical-build evidence. Its exact
same- and fresh-interpreter continuation checks and both complete test suites
passed. The historical intermittent SLEPc omission remains unexplained, and
the full calibration remains incomplete. The earlier library-identity correction
and smooth/sharp study machinery are committed at `42b9b71` and `cd998f6`,
respectively. Their evidence and the current qualification limits are below.

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

Independent review subsequently identified a missing finite-field guard before
sensor interpolation. The guard correction at `40f1324` passed 95 posterior,
JAX-connection and dynamics checks in 6.80 s; both scoped reviews approved it.
The earlier 456-test suite and original-environment checks precede that guard
change. The completed pilot below used `40f1324`; all 100 calibration experiments
remain required.

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

## Reference-informed public controls

At source `40f1324`, all three mesh-16 spread pilots (0.1, 0.3 and 1.0)
passed the predeclared criteria using four chains, 256 warmup and 256 retained
trajectories per chain. Recorded sampling times were 4141.13, 3953.08 and
3936.37 s respectively. The selected spread is 1.0, the largest passing choice.
The complete three-mesh pilot finished under that source; these are numerical
pilot results, not evidence of production convergence or completed calibration.

The preserved qualification record is
`runs/hmc-pilot-centered/qualification.json`, with manifest identity
`dcb1284ed35e8f25ab0c702469c5975608e6b06e87bdd8d7462c4e79e0893b27`.
For the selected spread 1.0, every chain passed the declared spread criteria
and had zero retained divergences:

| Mesh | Sampling time (s) | Minimum pilot ESS | Worst pilot R-hat | Frozen `(rtol,atol)` |
| --- | --- | --- | --- | --- |
| 16 | 3936.374 | 443.707 | 1.012507 | `(1e-10,1e-12)` |
| 32 | 7330.666 | 466.035 | 1.016126 | `(1e-10,1e-12)` |
| 64 | 26290.458 | 440.276 | 1.017080 | `(1e-12,1e-14)` |

Preliminary force checks passed tolerance indices `[1,2]`, `[1,2]` and `[2]`
respectively; checks on ten actual retained states passed `[0,1,2]` at each
mesh. Indices refer to the ordered tolerance pairs in the numerical-pilot
procedure below. The production request begins with 1000 warmup and 2000
retained iterations per chain. Pilot R-hat values exceed the production
threshold 1.01, so these short chains remain production-unqualified.

Before the new defaults, the shared-target, JAX, dynamics, Gaussian and complete
continuation checks passed 150 tests in 539.29 s. The final process orchestration
and study-helper checks passed 39 tests in 31.64 s, with three known direct-LU
unused-option notices. After the defaults change, the real serial/concurrent/
interrupted heat-process check and reference-default checks passed four tests
in 105.12 s. These verify identical saved physical histories for ordinary versus
explicit controls and interrupted serial versus uninterrupted concurrent toy
calculations. Logs are `task-13-shared-target-production-green-continuation.log`,
`task-13-concurrency-combined-green.log` and
`task-13-concurrency-native-defaults-green.log` in the task evidence directory.
At combined revision `721f303`, the full qualified suite finished with 530
passed tests, four passed subtests and one failure in 1470.55 s. The failure
was a stale invalid-control test: its original large spacing bound had become
implicit after the public defaults changed. The test now declares its original
spacing and bound, preserving the prior-resonance rejection criterion.

Review also identified missing admission control after Python memory exhaustion.
A subsequent check exposed parent inspection briefly taking the replica lock
before the issued child could acquire it. Deterministic child-process regressions
confirmed both failures. The corrections preserve unresolved evidence, stop
admission after memory exhaustion, and reserve pending starts without contending
for their initial ownership lock. Terminal and continuation checks remain locked.

After these fixes, the execution, study-helper, dynamics and actual heat-process
checks passed 93 tests in 103.92 s, with three known direct-LU unused-option
notices (`task-13-combined-fix2-final-green.log`). The original environment's
execution checks passed 20 tests in 2.90 s
(`task-13-combined-fix2-final-original.log`). Its full suite before these fixes
passed 362 tests and four subtests, with 24 intentional optional skips, in
620.50 s. The earlier full qualified suite is not claimed as a post-fix pass.
Both independent scoped reviews approved these corrections at `7196c0a`.
The matching numerical pilot has completed; all 100 calibration experiments
and scientific evidence review remain required before Task 13 is complete.

Public `HMCControls` now uses initial spacing `6.5e-4`, bounds
`(1e-6, 1.3e-3)` and spread `1.0`. These settings combine the measured local
curvature with actual reference trajectories. The duration law remains
`(pi/4, 3*pi/4)`, the momentum covariance remains identity, and only spacing
adapts during warmup. Default spacing retains an implicit-provenance marker:
zero warmup still needs an explicitly supplied spacing and disabled adaptation.

The evidence is conditional on the reference prior, observations and meshes;
these controls are not a universal stability guarantee. Explicit study controls
and the notebook's initial spacing `5e-4` and ceiling `1e-3` remain unchanged.
The final combined numerical/study revision used a new matching pilot before
SBC. Its source identity cannot append to the earlier pilot's chains.

## Matching numerical pilot and calibration progress, 2026-11-05

The complete pilot at `7196c0a` used the same fixed physical observations,
prior, seeds and four-chain lengths as the earlier pilot. Each row below
contains four chains with 256 warmup and 256 retained iterations per chain.
Every declared spread criterion passed, with no failed PDE solves or retained
divergences. Spread 1 was selected. The frozen `(rtol,atol)` pairs are
`(1e-10,1e-12)` at meshes 16 and 32, and `(1e-12,1e-14)` at mesh 64.

| Mesh | Spread | Sampling seconds | Minimum ESS | Worst R-hat |
| --- | --- | --- | --- | --- |
| 16 | 0.1 | 2295.81 | 514.34 | 1.014615 |
| 16 | 0.3 | 2341.87 | 558.31 | 1.015898 |
| 16 | 1 | 2253.60 | 443.71 | 1.012507 |
| 32 | 1 | 5333.42 | 466.04 | 1.016126 |
| 64 | 1 | 23744.09 | 440.28 | 1.017080 |

These short pilot chains remain production-unqualified because their worst
R-hat exceeds 1.01. The initial production request is 1000 warmup and 2000
retained iterations per chain, subject to actual diagnostics and continuation.

A read-only comparison against `40f1324` found exact equality of all stored
physical fields, latent coordinates, sensor predictions and trajectory statistics
for all 10240 pilot trajectories (12929843 integration steps). This verifies
these five fixed-seed histories only. The mesh-16 spread-0.1 calculation took
2295.81 s versus 4141.13 s previously, a measured factor 1.80; system load was
not controlled, and the timing is not a universal performance guarantee.

Actual command:

```bash
python scripts/run_hmc_study.py pilot --config runs/hmc_reference_parallel.json --output runs/hmc-pilot-shared-target
```

The run configuration differs from `validation/hmc_reference.json` only in
`pilot_output`, set to `runs/hmc-pilot-shared-target` before the experiment.
The immutable pilot manifest is `runs/hmc-pilot-shared-target/manifest.json`,
with identity `cf42944b4dccf8a03273054ed88700268ff48cba6fef413a6cc1ca1817a61d42`.
The qualification record is `runs/hmc-pilot-shared-target/qualification.json`,
whose SHA256 is `59dfda67e0778d5fd07e64e0e2fea5346bcb8d067bd86a5a2732c1071a2e7f21`.
Together these records contain the actual numerical environment, force
comparisons and chain checksums. The full comparison script and JSON are preserved as
`task-13-all-pilot-histories.py` and `.json` in the task evidence directory.

After the pilot completed and its identity/checksums were verified, the fixed
100-index calibration experiment began with two independent workers:

```bash
python scripts/run_hmc_study.py sbc --config runs/hmc_reference_parallel.json --output runs/hmc-sbc-shared-target --replicates 100 --workers 2
```

Before the cooperative stop on 2026-11-05 at 16:13 UTC, replicas 0 through 9
each qualified with four chains and 2000 retained draws per chain. Replica 10
encountered a heat-solve failure during the second chain's warmup: PETSc reason
`-9`, nonfinite residual and zero solver iterations. Its first chain had completed
1000 warmup and 2000 retained iterations. The generated truth, data, failed
position and all committed work are preserved for diagnosis. Replicas 11 and 12
closed cooperatively at saved trajectory boundaries. This partial campaign does
not support a calibration conclusion.

The first two posterior ranks are `[72,21,11,83,20]` and `[55,90,18,49,71]` in
the declared five-quantity order. All original indices, truths, observations,
chains, failures and continuation records remain part of this preserved
100-experiment study. Its manifest identity is
`1fd2bddfc2ab4fcb3258d1f8e7fb91d2c78989b0d27d7de92282874eff44821d`.

## Failed-warmup diagnosis and candidate spacing, 2026-11-06

Evidence names in this section and the next are relative to
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/`.

The immediate fatal arithmetic is now localized: the failed field has positive,
finite conductivity `2.10245e249` to `3.21445e306`, but its assembled heat matrix
and source vector contain nonfinite entries. This establishes assembly overflow
before an iterative-solver failure alone could explain the event; the individual
overflowing assembly operation has not been isolated. A failed heat evaluation
at a finite field still stops the calculation and preserves its evidence.

The Hamiltonian error grows much earlier, while conductivity is ordinary. An
exact saved-key replay reproduced the first returned state and endpoint energy
error, and the next trajectory's failed position. The first trajectory, with
spacing `0.00065`, ended at `Delta H=-15684.024`. Its ordinary Metropolis
acceptance is one because energy decreased; step-size adaptation then increased
the next spacing to its `0.0013` cap. That next trajectory already had
`Delta H=-165286.496` after one step and a maximum absolute error `2.47843e7`
by step 12. At step 11, conductivity ranged only from `0.62446` to `1.37769`.
The eventual extreme-conductivity assembly overflow therefore follows an earlier
loss of Hamiltonian accuracy. Large negative energy errors remain divergences
even when ordinary Metropolis acceptance is one.

Two predeclared comparisons used the same second-trajectory start, momentum and
realized duration `1.7628`, with half and quarter spacings. Both completed,
although both remained divergent. Their maximum absolute energy errors were
`60509.967` and `15087.236`, a factor `4.0107` apart. A separate bounded
observation used the same short Hamiltonian time `0.0156` from each saved start:

| Spacing | Maximum absolute energy error, first start | Maximum absolute energy error, second start |
| --- | --- | --- |
| `0.000325` | 23863.157 | 14999.534 |
| `0.0001625` | 5956.920 | 3759.326 |
| `0.00008125` | 1488.637 | 940.345 |

The approximately fourfold reduction on halving spacing is the expected
second-order behavior. At the two starts and coarse-trajectory steps 6 and 11,
production evaluations also agree with independent direct-LU heat and adjoint
solves: maximum `abs(delta U)=5.97388e-5` and relative total/likelihood-force
discrepancy `2.33201e-11`. LU likelihood forces agree with centered directional
finite differences at displacements `2e-4` and `1e-4` to relative error at most
`4.84476e-9`. Forward/reverse relative position and momentum errors are below
`1.75e-14` and `3.13e-11`. Reversibility alone would not validate a force; the
independent force comparisons and spacing convergence together support inadequate
Hamiltonian time resolution as the explanation of this observed escape.
These measurements concern the observed fields and trajectories and do not
establish stability for every prior-predictive realization.

The read-only diagnostics preserve all protected input hashes and add no draws
to a posterior. Their evidence is `task-13-replica10-trajectory-probe/replay.json`,
`task-13-replica10-local-probe/local.json`, the adjacent compressed scalar
histories, and `nov6-hmc-replica10-followup-report.md`. The fixed-point assembly
observation is in `task-13-replica10-fixed-point-probe/probe.json`.

### One declared candidate; pilots running

The declared candidate uses initial spacing `0.0001625` and bounds
`[1e-6,0.000325]` in an explicit `StudyConfig`. Public defaults remain
`0.00065` and `[1e-6,0.0013]`. Target acceptance `0.8`, identity momentum,
the original duration law, exact Gaussian rotation, ordinary Metropolis decision
and step-size-only BlackJAX adaptation remain fixed. The candidate is a proposed
pilot setting; even its short-time errors above can exceed the divergence
threshold at dispersed starts. Actual settled-warmup and retained-trajectory
criteria determine qualification.

On 2026-11-07, both declared full three-mesh pilots were launched from source
`4853edc`, on the preserved replica-010 observations and the original analytic
reference observations. Both use `runs/hmc_resolution_candidate.json` and new
output directories. Their commands are:

```bash
python scripts/run_hmc_study.py pilot --config runs/hmc_resolution_candidate.json --output runs/hmc-pilot-eps1625-replica010 --data runs/hmc-sbc-shared-target/replicate-010/generated.npz
python scripts/run_hmc_study.py pilot --config runs/hmc_resolution_candidate.json --output runs/hmc-pilot-eps1625-analytic
```

Both pilots are running; neither is complete or qualified at this report update.
The replica-010 pilot has finished its initial mesh-16, eleven-state,
three-tolerance comparisons and entered spread-0.1 warmup. The retained-state
force checks remain required. Their logs are
`task-13-resolution-4853edc-{replica010,analytic}.log` under the evidence root
above. The configuration and replica-010 observation hashes remain unchanged.
Each pilot must compare the original mesh-16 spreads `0.1,0.3,1.0`, with four
chains, 256 warmup and 256 retained iterations, and select the largest spread
passing the unchanged rules.
Every failed spread remains visible. Force tolerances must be requalified on
each mesh and actual data set. Any disagreement between the two spread choices
requires an explicit resolution before fixing campaign-wide initialization.
There is no declared fallback candidate. Actual solve counts, timing, acceptance,
energy errors and ESS must be measured; nominal fourfold integration work at
these smaller initial/cap settings is not a prediction of ESS per unit time.
The unchanged public defaults can be reconsidered only after measured pilots.

The stopped campaign retains its original numerical identity, truths, data,
chains and failed/interrupted outcomes. Changed source or controls require new
numerical identities and outputs. No old chain is migrated, no difficult truth
is replaced, and these diagnostic observations cannot complete calibration.

## Reviewed corrections and current verification, 2026-11-07

Broader validation had exposed a separate library-identity false rejection:
a captured NumPy library appeared under two sibling environment paths, with
the same device, inode, size and SHA256. Both names were hard links to one file;
all 188 library files recorded by the stopped campaign were unchanged.
The correction committed at `42b9b71` accepts an alternative sibling-installation
name only when the same relative path under the active interpreter's installation
identifies the same file and both byte hashes agree. Genuinely distinct files,
changed bytes and ambiguous library names remain rejected. The new identity
records schema 2 and its explicit path policy; source/build changes still prevent
appending to an old calculation. Both independent scoped reviews approved this
correction.

Thirty fresh identity observations in each numerical installation completed
with stable identities. Neither repeat actually encountered an alternative name;
this is measured repeatability evidence only. Separate regression checks use
real temporary hard links and files to exercise alias acceptance and strict
rejection of distinct or altered files. The final focused qualified-installation
checks passed **55 tests and 15 subtests in 571.88 s**; the original installation
passed **25 tests and 15 subtests in 62.16 s**, with no failures, errors or skips
in their XML. The evidence is `task-13-native-fix3-{qualified,original}.{log,xml}`,
`task-13-native-corrected-{qualified,original}-01/complete.json`,
`task-13-native-fix3-report.md`, and
`task-13-native-fix3-{spec,standards}-review.md`. These checks do not qualify the new spacing candidate.

The smooth/sharp study machinery committed at `cd998f6` also passed both scoped
reviews. It preserves the previous complete trajectory while preparing an
extension, protects saved outcomes while a calculation is active, and reduces
failed trajectories to the five physical scalar histories without retaining
all full conductivity fields in memory. The reviewed combined focused checks
passed **55 tests in 61.005 s**, with no failures, errors or skips. Evidence is
`task-14-fix2-native.{log,xml}` and
`nov6-hmc-task14-fix2-{spec,standards}-review.md`.

The complete combined-source checks have now finished:

| Numerical installation | Complete-suite result | Elapsed seconds |
| --- | --- | --- |
| Qualified | 582 passed, 19 subtests passed, **1 failed** | 1507.36 |
| Original | 412 passed, 19 subtests passed, 26 optional skips | 622.12 |

The qualified failure is
`test_changed_identity_rejected_before_target_extension[source_hash]`: its
initial sample is rejected because two numerical identities differ, before the
intended source-change check. The saved `problem.json` contains 187 selected
libraries, whereas the independently recorded SQLite environment contains 188.
The sole addition is `libslepc.3.25.1.dylib` and its Conda ownership record;
all other environment fields and common library byte hashes are identical.
Recomputing both numerical identities reproduces the saved and failing
identities exactly. The physical problem identity is unchanged and no initial
trajectory was committed. This isolates the recorded discrepancy, but the cause
of SLEPc's intermittent inclusion remains unmeasured.

The recorded identity evidence contains 484 stable observations, including the
captures from the three isolated executions of the failing test. They have
not reproduced or explained the historical SLEPc omission. That complete
qualified suite did not pass; the later correction and its separate qualification
are described below. The full-suite logs and XML are
`nov6-hmc-consolidated-{qualified,original}.{log,xml}`. A strict documentation-only
Sphinx build passed (`nov6-hmc-consolidated-docs.log`); that check precedes this
report update.

### Figure production and a fixed numerical experiment

A public fixed-field calculation followed by notebook-style PNG production
left conductivity, temperature, sensor predictions, potential, likelihood and
both total and likelihood forces exactly unchanged. The old comparison still
rejected the subsequent calculation because distinct SciPy/Matplotlib extensions
and a Pillow-private helper shared library basenames. The corrected comparison
uses observed package origins, unique file ownership, actual paths and byte
hashes to distinguish these components. The fresh numerical reference remains
fixed independently of figure production or the parent's import history.
Additional numerical components must belong to that same observed installation;
foreign numerical builds, changed core libraries and unresolved origins reject
continuation. This bound concerns the measured installation and recorded loader
controls; it does not establish every possible dynamic-loader behavior.

Separately controlled comparisons reproduced a missing SLEPc classification and
creation of a calculation with disagreeing declared and observed numerical
evidence. The candidate includes SLEPc in the strict comparison and checks both
records before declaring creation complete. Failed or interrupted creation keeps
the declared and observed evidence distinct and cannot be reopened as a complete
calculation. These corrections do not identify the cause of the historical
187-to-188-library omission. The final candidate also reduces repeated inspection
bookkeeping while renewing the installation evidence at every observation;
the conductivity posterior, force and trajectory rules remain unchanged.

The final candidate passed all 13 steps of the public calculation, PNG and
continuation experiment. Intentionally interrupted small heat calculations resumed
after plotting in the same interpreter and after load and plotting in a fresh
interpreter. Both matched the uninterrupted calculation exactly in all saved
warmup and retained records, initial/final checkpoints, identities and
controls, excluding only measured trajectory wall time. All fixed-field physical
comparisons and caller-field preservation were exact. These short calculations
test reproducibility and continuation; they supply no new convergence, mesh
accuracy or calibration evidence. Both scoped independent reviews approved the
bounded correction and the final inspection-only reduction.

| Numerical source | Installation | Complete-suite result | Elapsed seconds |
| --- | --- | --- | --- |
| Earlier candidate (round 4) | Qualified | 613 passed, 32 subtests, zero skips | 4160.85 |
| Final source `4853edc` (round 5) | Original | 443 passed, 32 subtests, 26 expected optional skips | 1109.43 |
| Final source `4853edc` (round 5) | Qualified | 614 passed, 32 subtests, zero skips | 2551.45 |

The final qualified and original XML records contain 646 and 501 cases,
respectively, with zero failures or errors. Both the earlier and final qualified
runs emitted three existing Gaussian-test warnings about `record_property` and
the xunit2 JUnit report format. The earlier result applies to its exact source.
The final source's strict documentation build passed before this prose update.
The exact reviewed correction was committed at
`4853edcfebe6d22d5f2f64acb2d87c1e8989f696` on 2026-11-06. Candidate evidence,
including earlier results, scoped reviews and the final public experiment, is in
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/nov6-provenance-fix5-candidate/`.
Final complete-suite, documentation and application evidence is in
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/nov6-provenance-fix5-qualified/`.
The two running three-mesh smaller-spacing pilots use this frozen numerical
source. The public default controls remain unchanged.

Tasks 13 and 14 remain scientifically incomplete. Task 13 still requires
completion of the matching pilots, a qualified fixed-100 calibration and
scientific evidence review. Task 14 still requires refined smooth/sharp truth observations, all six
baseline and eight prior-sensitivity posterior outcomes, and inspection of the
physical figures and notebook. Passing focused checks or reviewing study
machinery does not supply those physical results.

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
The default runs one predeclared replica at a time; `--workers` explicitly bounds
simultaneous independent replicas as described below. The fixed indices and seeds
are preserved for either schedule. Repeated one-replicate commands would repeat
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
| Numerical correctness of fixed discrete posterior evaluations | Three-mesh pilot complete at `7196c0a`; matching smaller-spacing pilots running at `4853edc`, not yet qualified |
| Sampling convergence | Ten calibration posteriors qualified, one warmup failed, two interrupted; study incomplete |
| Mesh approximation to the physical solution | Reviewed study machinery exists; refined truths and physical outcomes remain pending |
| Sharp-truth prior adequacy | Baseline and prior-sensitivity outcomes remain pending |

The required 100-replication calibration remains incomplete. This report will
be updated with actual source/environment records, commands, figures,
measured costs, failures, outcomes and immutable manifest/chain checksums
after the controller executes and reviews the scientific studies.

## Independent calibration processes

`run_sbc(config, output, replicates=100, workers=1)` and the corresponding
`--workers` CLI option bound the number of independent replicated experiments
running together. The default remains one. Every replica uses a fresh Python
process and its own four chains. Changing worker count changes execution
provenance, while the fixed replica indices, four distinct random streams per
index, discrete prior and heat model, duration law, chain lengths, convergence
extensions and rank diagnostic remain unchanged. There is no automatic resource
scheduler. A worker count must be an explicit integer between one and the fixed
replica total.

The parent writes aggregate progress, calibration figures and execution records.
Each replica owns its local events, log, generated truth/data, mode and posterior.
A durable issued-attempt record precedes process launch; the replica lock precedes
native verification and remains held through terminal publication. Restarting the
parent cannot create a second writer for a child that survived it. An issued but
unclosed attempt reserves capacity even when no lock is currently held: it could
be waiting to acquire the lock, or its death could remain unexplained.

Only an explicit cooperative interruption, recorded after safe cleanup, permits
continuation. It reuses the original truth, observations, saved chains and random
state, and preserves the largest saved extension target. A completed scientific
failure remains in the experiment; an unexplained exit remains unresolved across
restarts. Neither causes a replacement truth draw. Finite insufficient ESS or
R-hat continues the same chains under the existing extension rule. Missing or
nonfinite evidence and retained divergences require investigation. Known
system-wide resource failures stop admission of additional experiments.

Execution histories live under `executions/`; replica-local `attempt-*.json`,
`attempt-*.log`, `events.jsonl` and `result.json` retain ownership and outcome
records. A valid terminal result can survive loss of its separate closure record;
its scientific checksums are verified before aggregation. All numerical and study
sources, including the worker, remain bound to the study identity. Adoption
therefore requires a new matching numerical pilot before SBC. The short process
tests exercise ownership and actual heat sampling only; they provide no physical
calibration or production convergence evidence.
