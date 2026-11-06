# Conductivity posteriors across meshes and sharp interfaces

**Status: study apparatus prepared; physical results pending.** No smooth/sharp
fine-truth record, six baseline posterior calculations, or eight prior-sensitivity
calculations is claimed here. Task 13 numerical qualification and its 100
same-model calibration experiments remain separate requirements. Passing execution
checks does not establish continuum accuracy or prior adequacy.

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

## Reproducible calculation sequence

Run these commands only after freezing the reviewed source and numerical
environment. Each configuration records its own `pilot_output`. A pilot using
another observation vector, prior, source or numerical environment cannot qualify
a mesh study. The data-specific pilot currently evaluates all three inference
meshes, including for a sensitivity configuration subsequently sampled at 16.

```bash
python scripts/run_hmc_study.py truth --case smooth \
  --config validation/hmc_reference.json --output runs/truth-smooth
python scripts/run_hmc_study.py truth --case sharp \
  --config validation/hmc_reference.json --output runs/truth-sharp
```

Prepare explicit copies of the reference JSON with `pilot_output` set to the
corresponding data-specific pilot directory. For example,
`validation/smooth-baseline.json` names `runs/pilot-smooth-baseline`. These files
must be saved before launching their pilot, then passed unchanged to the mesh
calculations.

```bash
python scripts/run_hmc_study.py pilot \
  --config validation/smooth-baseline.json \
  --data runs/truth-smooth/observations.npz --output runs/pilot-smooth-baseline
python scripts/run_hmc_study.py mesh --nmesh 16 \
  --config validation/smooth-baseline.json \
  --data runs/truth-smooth/observations.npz --output runs/smooth-n16-baseline
```

Repeat the mesh command at 32 and 64 with separate output directories, and repeat
the entire data-specific sequence for sharp truth. The baseline physical prior
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

Before Task 14 can be called complete, save and inspect both refined truth
records, all fourteen outcomes and their data-specific numerical qualifications;
resolve or explicitly retain every failed scientific criterion; inspect the
notebook/report figures; pass the ordinary regression checks and warning-free
documentation build; and complete the planned independent review.

## Explicit extension and preserved analysis generations

If ordinary chain diagnostics qualify but contiguous-block uncertainty remains
unresolved, request a larger retained total per chain explicitly:

```bash
python scripts/run_hmc_study.py mesh --nmesh 16 \
  --config validation/smooth-baseline.json \
  --data runs/truth-smooth/observations.npz --output runs/smooth-n16-baseline \
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
