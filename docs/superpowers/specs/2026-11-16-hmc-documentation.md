# Documentation requirements for conductivity optimization and sampling

Date: 16 November 2026. This document translates the user's documentation request and accepted HMC scientific contract into the scope of the documentation update. It does not change the statistical model or authorize a new inference campaign.

## Purpose and audience

A reader should understand how noisy thermometers constrain a spatial conductivity field, obtain either an optimized estimate or posterior samples from one declared experiment, and interpret uncertainty without mistaking convergence for accuracy or calibration. Explain the physics first, then the calculation interface.

Primary references are the [accepted full-field design](2026-10-16-full-field-hmc-design.md), the 17/18 October interviews, ADRs 0001–0003, and the [16 November review](../reviews/2026-11-16-hmc-review.md). The reviewed implementation is `3df71cacf650f94d77c54fc0dbcf0f916825e199`; later numerical qualification evidence supersedes provisional defaults in the original design.

## Required scientific explanation

- Infer every conductivity degree of freedom on the selected P1/Q1 mesh from fixed physical temperature sensors. Preserve source, boundaries, sensor positions, readings and errors across the experiment.
- State the independent Gaussian sensor likelihood and proper Gaussian prior on log conductivity. Define the whitened coordinates, the potential and positivity transformation. Do not introduce an extra conductivity-density Jacobian.
- The new optimization and sampling interfaces use the same posterior. Explain that the optimized result is its mode in log conductivity and the plotted sampling estimate is the mean of physical conductivity. The legacy TAO reconstruction uses its original, different regularized objective.
- Explain the discrete adjoint force: one heat solve and one adjoint solve per likelihood-gradient evaluation, independently of the number of conductivity unknowns. This is not a claim of two solves per HMC trajectory or sample.
- Describe Gaussian-prior rotations, likelihood kicks, independent chains, randomized trajectory durations, frozen retained-sampling controls and step-size-only adaptation. BlackJAX supplies component machinery; this is a custom split-Hamiltonian calculation, not an unmodified default NUTS or stock HMC invocation. The PDE is evaluated through its existing adjoint connection in eager CPU float64 mode.
- Distinguish pointwise credible intervals, Monte Carlo errors, repeated-experiment calibration, mesh discretization errors and prior adequacy. Gaussian priors can smooth sharp interfaces; infrastructure support is not a promise of exact recovery.

## Required workflow

Cover explicit observations/prior/numerics, construction of `HeatInverseProblem`, `optimize`, complete-state initialization of `sample`, matched-probability `diagnostics` and `summary`, saved physical fields, `PosteriorSamples.open`, `HeatInverseProblem.load`, `resume(total_draws=...)`, explicit `apply_to` and deterministic closure. State which operations require PDE solves and which can read saved arrays without MPI or sampling packages.

Examples must be internally complete or clearly identify inputs supplied by an immediately preceding block. Keep a small executable illustration explicitly unqualified. State actual defaults separately from example settings. Show failed optimization and stale/incomplete diagnostics honestly. Closing an inverse problem does not transfer ownership of the caller's forward solver.

Continuation requires matching physics and numerical controls. Explain the newly discovered form-compiler identity limitation until its correction is independently verified. Ordinary notebook reports must use the current study generation and the detected project root after the reviewed defects are fixed.

## README and figure requirements

- Use “Sampling” and “Optimization” in figure legends. Explain HMC in the README prose.
- Use the physical-conductivity posterior mean and equal-tailed pointwise 68% and 95% intervals. Use one sampling color with two band opacities and one distinct optimization color.
- Keep friendly axis labels and a concise description. Avoid additional panel prose and diagnostic overload.
- Reuse the prepared central horizontal/vertical conductivity slices, exported values and provenance. They represent four mesh-32 chains with 1,024 retained draws each and a generating mesh of 16. Label the example accordingly; it is not the finer-truth mesh-refinement study.
- Keep the main README figure focused on recovered conductivity and uncertainty. A spectral figure is optional and deferred until its transform, boundary treatment, conjugate weighting and intended estimand are independently verified. No Hann-window claim is inferred from the current review.
- Do not recompute the posterior, change priors, pool different experiments or alter saved scientific records to make the figure more attractive.

## Global constraints

- The current sampling qualification uses CPython 3.12, CPU float64, one MPI rank, BlackJAX 1.7.1, JAX/JAXlib 0.11.2 and ArviZ Stats 1.3.3.
- Preserve the native conda finite-element environment; inspect constrained optional-package resolution before installation.
- Preserve flat top-level imports, the existing legacy reconstruction semantics and the caller's supplied physical state.
- Use “Sampling” and “Optimization” in figure legends. Explain HMC in the README prose.
- Posterior uncertainty bands are equal-tailed pointwise 68% and 95% credible intervals of physical conductivity.
- Four-chain qualification requires R-hat < 1.01, bulk ESS >= 400, tail ESS >= 400 and no retained divergences for the monitored quantities; convergence alone does not establish calibration or continuum accuracy.
- Preserve immutable study inputs and saved chain histories. Documentation work must not launch new production studies.
- Do not edit the frozen calibration source tree. Implement documentation in an isolated working tree based on the reviewed implementation and any subsequently verified corrections.

## Evidence and delivery

Use a dated validation summary with source revision, numerical environment, completed checks, incomplete studies and explicit links to public evidence. Local `.superpowers` or `/private/tmp` paths must not appear as readers' required inputs. Publish small stable numerical summaries/provenance when needed; do not add chain databases to documentation.

The README and Read the Docs landing page must both present optimization and sampling. Organize tutorial, theory, practical recovery guidance, API reference and validation evidence consistently. Preserve useful existing links and legacy examples. Update stale test counts and notebook execution descriptions from the actual current tests rather than copying old prose.

Acceptance requires a warning-free Sphinx build, resolved local links/images, inspected rendered pages, execution of the bounded example in the declared environment, and explicit limits on any checks not completed. A mocked documentation build does not validate native execution. Record deployment separately from a successful local build.

The review identifies corrections needed before some claims can be strengthened. This documentation plan may record those limitations while fixes are pending; it must not silently treat them as resolved. Any source corrections belong to a separately reviewed implementation change.
