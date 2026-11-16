# Conductivity posteriors across meshes and sharp interfaces

**Status, 16 November 2026: reference observations qualified; production mesh
and prior studies incomplete.** The six baseline and eight prior-sensitivity
outcomes below remain outstanding in the available evidence. The qualified
four-chain mesh-32 slice demonstration used a generating mesh of 16; it does
not complete this finer-truth study. The current saved same-model calibration
record has 8/100 qualified replicas and remains incomplete. See the
[dated scientific evidence summary](hmc-status.md) for the separate scopes.

## Historical status, 11 November 2026

**Status, 11 November 2026: reference observations frozen; production studies
pending.** Both reference records meet the declared sensor-refinement criterion.
The smooth/sharp baseline pilots and Task 13's two smaller-spacing pilots wrote
their numerical completion records before reporting MPI shutdown errors. The
saved-chain audit found intact records; the exploratory posteriors do not
establish production convergence. See the separate
[shutdown qualification](hmc-mpi-shutdown.md). Prior-sensitivity pilots, all six
baseline posterior calculations, eight prior-sensitivity calculations and the
100 same-model calibration experiments remain incomplete. These are separate
scientific requirements.

The heat source is 1 on the unit square, the bottom temperature is 300 K and
other boundaries are insulated. The same 256 interior thermometers at
`(i/17,j/17)` have independent known errors `sigma=0.001 K`. Every conductivity
node is inferred: 289, 1089 and 4225 unknowns at `nmesh=16,32,64`.

The smooth truth is `h=1+0.35*x+0.25*y+0.15*x*y`. The sharp truth uses the existing
`_sharp_interface_h`: conductivity 1 where `x+0.25*y<0.62`, and 3 elsewhere.
Each is represented by continuous nodal Q1 conductivity on the truth mesh.
Reference temperatures are first compared at 128 and 256. Further doublings
continue until the maximum sensor-temperature difference is strictly below
`0.1*sigma`; every comparison is recorded. Failure leaves reference qualification
open. This is a sensor refinement criterion, rather than an exact continuum-error
bound for conductivity.

Only then is one noise realization added, using seed 20261026 for smooth and
20261027 for sharp. The archived arrays contain `points`, `values`, `sigma` and
`noiseless`, with a checksum, truth definition, source/boundaries, noise seed,
finer mesh, and refinement evidence. Every inference mesh and sensitivity case
reads those same bytes. This experiment is separate from same-discrete-model
simulation-based calibration.

## Frozen reference observations, 2026-11-07

Both truth commands below exited successfully at checkout revision
`fcb7466782e3699e23720073819d606367aa0361`. Every numerical source remained
unchanged from `4853edcfebe6d22d5f2f64acb2d87c1e8989f696`. Direct-LU heat
solutions gave the following maximum change over the 256 fixed sensors:

| Truth | Coarse mesh | Fine mesh | `max(abs(T_fine-T_coarse))/sigma` |
| --- | --- | --- | --- |
| Smooth | 128 | 256 | 0.00838298677763305 |
| Sharp | 128 | 256 | 0.34993962077578544 |
| Sharp | 256 | 512 | 0.14809598923193334 |
| Sharp | 512 | 1024 | 0.0774712140128031 |

The smooth record therefore uses mesh 256 and the sharp record mesh 1024.
The last comparison in each case is below `0.1*sigma`. This qualifies the
reference at the sensors under the stated refinement rule; it supplies neither
a rigorous continuum-error bound nor evidence of conductivity-jump recovery.
The sharp conductivity remains a continuous Q1 nodal interpolant with transition
cells. No posterior samples are produced by these truth calculations.

Each accepted reference received exactly one Gaussian-noise realization, with
seed 20261026 for smooth and 20261027 for sharp, at `sigma=0.001 K`.
Independent checks confirmed finite arrays, the fixed sensor coordinates and
ordering, and the exact seeded noise. The frozen outputs are
`runs/hmc-truth-4853edc-smooth/` and `runs/hmc-truth-4853edc-sharp/`.
The following SHA256 values identify the complete files, including each
manifest's observed numerical environment and source hashes:

| File in each output | Smooth SHA256 | Sharp SHA256 |
| --- | --- | --- |
| `observations.npz` | `7613edb728fd8172df56e687e27c8f05fad9496bc46a4c78ff292afd7810d8b6` | `469a4b597f0be81deb2a77c38125437db8d80125abaf1825ae07619fdba42c72` |
| `observations.json` | `f5722434857b3dd894f7618b65aabf08fa9065736347a7de6ba96b779e096d43` | `88db32d95ac587f4bb5ef1b3546489c2f170d9dc90626c2b0ece631ff2b4e64c` |
| `refinement.json` | `02fab60166f95400c7c00055d5e94860a7f69f8949062762b9ee99ee10cefdbd` | `d39dc945744b1204582440e8ff48e595977aab86ea2d86ea139ef91e972bc45e` |
| `manifest.json` | `971584527604d513c0bdfcc2f4c570cf2055aa766b50229ef3231a2d247e096d` | `a6c5762d942ed40f759fc14fd723722a52e10226645be45c6316b4dd02ebff32` |

Both commands used the qualified interpreter
`/Users/nguyenmn/miniconda3/envs/s2dhi-hmc/bin/python` and
`runs/hmc_resolution_candidate.json`, SHA256
`282e7f7d776dd317c9b6d1bf50778337549a6247783a4386d08d3e8d5e7dc36f`.
The invocation records, `truth-{smooth,sharp}.log`, execution outcomes and
`truth-verification.json` are preserved under
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/nov7-progress-and-truths/`.
Both logs include three known notices about unused iterative-solver options
in these direct-LU solves; neither command reported a failed heat solve.

## Reproducible calculation sequence

Run these commands only after freezing the reviewed source and numerical
environment. Each configuration records its own `pilot_output`. A pilot using
another observation vector, prior, source or numerical environment cannot qualify
a mesh study. The data-specific pilot currently evaluates all three inference
meshes, including for a sensitivity configuration subsequently sampled at 16.

```bash
python scripts/run_hmc_study.py truth --case smooth \
  --config runs/hmc_resolution_candidate.json --output runs/hmc-truth-4853edc-smooth
python scripts/run_hmc_study.py truth --case sharp \
  --config runs/hmc_resolution_candidate.json --output runs/hmc-truth-4853edc-sharp
```

Both actual-data baseline pilots were launched at `fcb7466` on 2026-11-07 and
have saved manifests. Their configurations differ from
`runs/hmc_resolution_candidate.json` only in `pilot_output`, naming the
corresponding data-specific pilot directory. The smooth configuration
`runs/hmc-eps1625-smooth-baseline.json` has SHA256
`8fe52f4ee6c94be47d9120d8de8549a17535cee1abfbe911f509a94279cee52d`;
`runs/hmc-eps1625-sharp-baseline.json` has SHA256
`ea4792fedd9963c14fbbad0ed8ab5bd1fa3175475907828a7bbedcf60ed1711e`.
The observation bytes are unchanged. The exact invocations and
`pilot-{smooth,sharp}-baseline.log` are recorded with
`truth-pilot-execution.json` under the durable evidence directory above.

```bash
python scripts/run_hmc_study.py pilot \
  --config runs/hmc-eps1625-smooth-baseline.json \
  --output runs/hmc-pilot-eps1625-smooth-baseline \
  --data runs/hmc-truth-4853edc-smooth/observations.npz
python scripts/run_hmc_study.py pilot \
  --config runs/hmc-eps1625-sharp-baseline.json \
  --output runs/hmc-pilot-eps1625-sharp-baseline \
  --data runs/hmc-truth-4853edc-sharp/observations.npz
```

Neither baseline pilot is complete or qualified. Their actual-data force checks,
settled warmup and retained-trajectory criteria remain required. No posterior
mesh-study outcome is claimed. After a matching pilot qualifies, its unchanged
configuration is used for the corresponding mesh calculations, for example:

```bash
python scripts/run_hmc_study.py mesh --nmesh 16 \
  --config runs/hmc-eps1625-smooth-baseline.json \
  --data runs/hmc-truth-4853edc-smooth/observations.npz --output runs/smooth-n16-baseline
```

Repeat the mesh command at 32 and 64 with separate output directories, and use
the corresponding sharp configuration and frozen observations for sharp posterior
calculations. The baseline physical prior
is `mu=log(2), kappa=2, tau=1, h_ref=1`, with natural Neumann boundaries. It is
identical across meshes.

Eight additional n16 outcomes are predeclared: for each truth, use `(kappa,tau)`
equal to `(1,1)`, `(4,1)`, `(2,0.5)`, and `(2,2)`. Retain `mu`, `h_ref`, and the
same frozen observations. Save each explicit configuration and qualify its
actual-data force calculations before sampling. Each change defines a separate
posterior and cannot be selected to force recovery of a known truth.

The ordinary sampler runs four dispersed chains, with 1000 warmup iterations
and an initial retained count chosen from the numerical pilot. Extensions
preserve all retained states and fixed dynamics. Missing/nonfinite diagnostics,
retained divergences or an integration spacing at its lower bound require
investigation. Failed calculations are retained; altered tuning starts a labelled
new directory with fresh warmup. No runtime or monotonic-acceptance condition is
introduced.

```bash
python scripts/run_hmc_study.py report \
  --manifest runs/smooth-n16-baseline/manifest.json \
             runs/smooth-n32-baseline/manifest.json \
             runs/smooth-n64-baseline/manifest.json \
  --output runs/hmc-mesh-report.md
```

Pass every available manifest, including failed and labelled replacement attempts.
The report always lists six baseline and eight sensitivity outcomes, marking
absent cases as missing. The public `write_study_report([], destination)` also
creates an explicitly empty evidence inventory. Completed reports link to actual
manifests, exact checksums, full diagnostics and physical figures.

## What the measured uncertainties mean

All-node rank-normalized split/folded R-hat, bulk/tail ESS, mean MCSE, interval-endpoint
MCSE and worst physical locations come from ordinary posterior diagnostics.
Qualification requires R-hat below 1.01, each monitored ESS at least 400 and zero
post-warmup divergences. Shared-location comparisons include the area mean and
conductivity at `(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)`, `(0.75,0.75)`.
Differences between meshes or priors are displayed with separate Monte Carlo
errors; these physical differences are not themselves sampling errors.

Acceptance and Hamiltonian energy distributions, within-chain autocorrelations,
actual heat/adjoint solve counts, retained trajectory times and sampling wall
time are recorded. Retained trajectory wall time excludes database commit and
progress-report overhead. Sampling wall time includes warmup, continuation and
storage overhead. Both are labelled explicitly; initialization and unsuccessful
work remain in the cumulative counters and event/failure records.

Efficiency uncertainty uses 500 seeded resamples of contiguous blocks within each
chain. It recomputes combined-chain bulk/tail ESS for the representative physical
quantities, the worst-ESS node, and the worst monitored observable (including
sensor predictions and spatial contrasts), with their paired measured trajectory
costs.
Block length is at least twice the largest measured correlation time, including
all-node diagnostic information and the measured acceptance, energy and cost histories.
The calculation combines total draws divided by bulk/tail ESS with within-chain
initial-positive-sequence autocorrelation-time estimates.
At least eight full blocks per chain are required. Insufficient blocks, missing
cost observations, or nonfinite resampled ESS leave uncertainty unresolved.
No artificial per-draw costs are inferred for older calculations that lack those
measurements. These intervals concern finite-chain uncertainty conditional on a
fixed posterior; they do not measure discretization or prior uncertainty.

Posterior predictive temperatures include independent observation noise. The
standardized sensor residual divides the posterior mean minus the observed
reading by `sqrt(Var(T|data)+sigma**2)`. The posterior predictive interval also
contains simulated measurement noise. The latent temperature spread remains
separately recorded.

## Physical interpretation near a material interface

Nodal 90% interval inclusion is reported separately within perpendicular distance
0.05 of `x+0.25*y=0.62` and outside that band. This is a spatial diagnostic from
one field and one noise realization. Perfect recovery and nominal 90% spatial
inclusion are not correctness criteria. Repeated-noise coverage would require
new independent noise realizations at fixed physical evaluation locations.

The figures show true conductivity, posterior mean, pointwise interval width,
and fixed-sensor predictive residuals. A continuous nodal field represents a
jump through transition cells. An exactly discontinuous material model may
therefore require a different field representation as well as a different prior.
Strong chain diagnostics alone cannot resolve this modelling limitation.

Before Task 14 can be called complete, save and inspect all fourteen outcomes
and their data-specific numerical qualifications;
resolve or explicitly retain every failed scientific criterion; inspect the
notebook/report figures; pass the ordinary regression checks and warning-free
documentation build; and complete the planned independent review.

## Explicit extension and preserved analysis generations

If ordinary chain diagnostics qualify but contiguous-block uncertainty remains
unresolved, request a larger retained total per chain explicitly:

```bash
python scripts/run_hmc_study.py mesh --nmesh 16 \
  --config runs/hmc-eps1625-smooth-baseline.json \
  --data runs/hmc-truth-4853edc-smooth/observations.npz --output runs/smooth-n16-baseline \
  --total-draws 8000
```

The requested total must exceed the saved total. This uses the ordinary
continuation mechanism with the same chains, dynamics and retained prefix.
Before any extension, the previous report and its checked physical/trajectory
artifacts are copied into `history/generation-NNNN`. A durable analysis request
records the new total and the preceding result checksum. An interrupted extension
continues that same request; changing its target during continuation is rejected.
Reports include archived generations as well as the current outcome. A failed
scientific calculation remains an investigation item and cannot be retuned by
this command.

`--reanalyze` recomputes summaries and figures from existing retained draws and
preserves the preceding generation. It performs no sampling and cannot by itself
create more independent blocks. It is mutually exclusive with `--total-draws`.

Result payloads and checksums are staged together in a recoverable publication.
The fine-truth qualification record is likewise checksum-protected, including its
refinement errors and a checksum of the refinement archive. Reuse rejects changed
qualification metadata as well as changed observation bytes.

For unsuccessful posteriors, available diagnostics, acceptance/energy
measurements, autocorrelations, elapsed sampling time, and solve counts remain
visible in the failure report. Undefined efficiencies remain explicitly
unresolved. Total solve costs include committed checkpoint counts plus every
recorded `aborted_work` increment once; retained-trajectory costs remain separate.

Figures are rendered in a fresh child interpreter. This keeps optional plotting
libraries out of the parent heat-calculation interpreter, preserving the native
library identity required by later calculations. True conductivity and posterior
mean use common color limits; interval width has its own physical scale.
