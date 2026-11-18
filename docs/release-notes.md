# Release notes

## 0.2.0 — Conductivity inference with uncertainty

The second public release adds full-field Hamiltonian Monte Carlo (HMC) sampling
to the steady heat inverse problem. Recover a conductivity field by optimization,
or explore its posterior distribution to estimate a mean field and pointwise
credible intervals. Both routes use gradients from the discrete adjoint method.

### New in this release

- A common conductivity posterior for optimization and sampling, with fixed
  physical thermometers, known independent Gaussian measurement errors and a
  proper Gaussian prior on log conductivity.
- A CPU, float64, single-rank HMC calculation using BlackJAX components, with
  Gaussian-prior rotations, likelihood kicks and a Metropolis correction.
- Saved physical-conductivity draws, convergence and Monte Carlo error summaries,
  and reproducible continuation of the same numerical experiment.
- A worked sampling notebook, posterior and adjoint derivations, installation and
  saved-result guides, and illustrations of the forward and inverse problems.
- Corrected MPI shutdown handling, continuation identity, resource-exhaustion
  handling, legacy initial temperatures, notebook reporting, optimization work
  counts and Fourier-mode weights.

The existing PETSc TAO reconstruction retains its original regularized objective.
It is distinct from the common posterior used by the new optimization and sampling
interfaces. Earlier multifidelity functionality is also included; measured work
counts do not establish an efficiency gain at a common converged solution.

### Installation and compatibility

Follow the [installation guide](installation.md). DOLFINx, PETSc and MPI require
the conda environment. Sampling is qualified with CPython 3.12, CPU float64 and
one MPI rank; the pinned sampling and diagnostic libraries are installed on top
of the native scientific environment.

Existing posterior archives remain readable. Continuation requires matching
sources and numerical settings, so archives created before the numerical-identity
corrections may need their original checkout and environment to extend a chain.

### Validation and scientific scope

The numerical code merged in [PR #2](https://github.com/MinhMPA/steady-2d-heat-inverse/pull/2)
passed 673 local tests and 32 subtests with no skips. Linux checks passed all
642 HMC tests without skips; the general environment passed 499 tests and 32
subtests with 29 expected optional skips. These selections overlap and are not
additive. [HMC checks on master](https://github.com/MinhMPA/steady-2d-heat-inverse/actions/runs/44240404104)
and [general and documentation checks](https://github.com/MinhMPA/steady-2d-heat-inverse/actions/runs/44240404054)
record the merged numerical baseline. The release preparation changes presentation
and version metadata, with no changes to the numerical sources or tests.

The saved mesh-32 example meets its declared convergence criteria for one discrete
posterior. Its 68% and 95% intervals describe pointwise uncertainty conditional on
the observations, prior and heat model. They do not establish simultaneous field
coverage, unbiased recovery or continuum accuracy. The recorded 100-replica
calibration and smooth/sharp mesh and prior studies remain incomplete. See the
[scientific evidence](validation/hmc-status.md) before interpreting a new physical
experiment.
