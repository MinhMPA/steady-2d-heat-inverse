# HMC requirements interview

Date opened: 2026-10-17.
Status: Q1–Q16 accepted on 2026-10-17. The follow-on
[implementation interview](2026-10-18-hmc-implementation-interview.md) records
Q17's selection of BlackJAX with the agreed split HMC on 2026-10-22 and the
accepted Q18–Q24 behavior. The consolidated specification is ready for final
shared-understanding review following Q24's acceptance on 2026-10-26.
The user requested an interview, with particular attention to fitting HMC
naturally into the existing conductivity-inference workflow.

## Confirmed requirements

- Infer the full conductivity field: retain every conductivity degree of freedom
  on the chosen mesh. Confirmed by the user on 2026-10-16.
- Discuss requirements from the physics and inference viewpoint first.
- Examine the proposed HMC calculation's relationship to the current solvers,
  observations, notebooks, and returned physical fields.
- Q1 accepted on 2026-10-17: one declared posterior for the new optimization and
  HMC workflows, with observations, noise model, and prior defined once.
- Q2 accepted on 2026-10-17: existing reconstruction calls retain their original
  scientific meaning and numerical behavior; the new posterior is selected
  explicitly. Q1–Q2 are recorded in `docs/adr/0001-share-the-conductivity-posterior.md`.
- Q3 accepted on 2026-10-17: validate on a small full-field mesh, then measure
  sampling efficiency at a representative workload. Q7 subsequently specified
  the resolutions and Q11 accepted one process per heat solve.
- Q5 accepted: both existing nodal observations and arbitrary fixed sensors,
  with known independent Gaussian errors and either common or individual
  standard deviations. Nodal observations retain their original coordinates.
- Q6 clarified: sharp conductivity variations are an eventual science case,
  and the calculation must permit other priors. The Gaussian baseline is
  acceptable without any expectation of perfect truth recovery. This is
  recorded in `docs/adr/0002-allow-priors-for-sharp-conductivity.md`.
- Q7 specified: test `nmesh=16, 32, 64`, corresponding to 289, 1089, and 4225
  conductivity unknowns on the current square meshes. Run lengths follow pilot
  timings; Q15 sets the diagnostic qualification criteria.
- Q4 accepted in the subsequent response to Q4/Q8–Q11: define one inference
  problem, optimize it, and pass the complete estimate into sampling. Direct
  initialization from a physical conductivity field also serves the baseline.
- Q8 accepted: return physical draws and their spatial association, posterior
  summaries, diagnostics, and access to sensor predictions; retain internal
  coordinates for continuation and future priors.
- Q9 accepted: new inference calls restore the supplied forward model; applying
  a selected field is explicit and recomputes its temperature. Recorded in
  `docs/adr/0003-preserve-the-supplied-physical-state.md`.
- Q10 accepted: initially fix the declared prior mean/correlation/amplitude
  parameters, check sensitivity through separate runs, and enforce positivity
  only. Do not infer those parameters or inherit numerical conductivity bounds.
- Q11 accepted: numerical checks followed by four independently seeded chains
  per mesh for smooth truth, then sharp-variation studies with the Gaussian
  baseline. Use one process per heat solve, measure cost and convergence on
  fixed observations, choose run lengths from pilot timings, and report
  unconverged cases explicitly. No specific runtime cap has been supplied.
- Q12 accepted: optimize the posterior density in log conductivity, label that
  choice, and return the corresponding physical field. Support uniform,
  callable, or mesh-associated physical initial fields without reducing the
  full-field dimension. Study chains have distinct seeds and dispersed starts.
- Q13 accepted: save draws and complete continuation states automatically,
  separate warmup, and validate the declared calculation on resume. Handled
  interruption preserves the last complete saved state and restores the supplied
  physical model. An unconverged PDE solve stops with evidence; nonfinite
  Hamiltonian proposals are rejected and counted.
- Q14 accepted: tune step size automatically during warmup, then hold it fixed.
  Use exact Gaussian rotations with likelihood kicks and randomized trajectory
  durations, optional explicit controls, fixed qualified PDE tolerances, and an
  established optional diagnostics package.
- Q15 accepted: four-chain rank-normalized split/folded `R-hat < 1.01`, combined
  bulk/tail ESS at least 400, and zero post-warmup divergences. Check all nodal
  conductivities and reported nonconstant physical quantities, report Monte
  Carlo errors, and retain flagged unconverged results.
- Q16 accepted: one fixed 16-by-16 interior sensor grid, known `sigma=1e-3 K`,
  and one finer-mesh data realization per truth case reused across all three
  inference meshes. Original nodal data remain a separate compatibility check.

The [consolidated specification](2026-10-16-full-field-hmc-design.md) incorporates
these decisions. Future prior examples remain possible extensions, not approved
first-release deliverables. This interview records requirements; no HMC
implementation or new sampling result is claimed.

## Evidence from the current calculation

- `src/tao_solver.py:250` computes an integrated squared temperature residual and
  conductivity roughness. This is a different objective from the proposal's
  fixed-sensor likelihood and log-Gaussian prior.
- `src/tao_solver.py:298` returns a conductivity array after updating the shared
  conductivity to the optimizer's solution. A field supplied to HMC as an
  initial condition need not be the mode of HMC's posterior.
- The original proposal preserved the original TAO objective without supplying
  optimization of the new posterior. Q1 resolved this gap: the revised proposal
  now requires optimization of the same posterior sampled by HMC. Q4 accepted
  the shared-problem calling sequence and complete-estimate handoff.
- `CONTEXT.md` already defines observations at fixed physical sensors, whereas
  the present adjoint consumes a temperature function or a nodal array. This is
  the distinction resolved by Q5's explicit nodal-observation conversion.

Additional facts checked during the interview:

- `notebooks/InverseSolve.ipynb`, cells 4–23 (zero based), loads nodal data,
  constructs forward and adjoint solvers, runs TAO, and immediately plots the
  returned conductivity array. The HMC call sequence and returned quantities
  are specified against this concrete workflow in Q4/Q8/Q9.
- The current adjoint includes the old roughness force and an integrated
  temperature residual (`src/adjoint_solver.py:95,128`); it cannot supply the new
  sensor likelihood's force unchanged.
- Existing noisy nodes can be treated as sensors at their original physical
  locations. Interpolating the noisy field to new positions generally produces
  correlated errors and cannot silently preserve the independent-noise model.
- A scalar conductivity creates a constant with no spatial unknowns in the
  current forward solver. The new physical-initial-field input must distinguish
  a uniform starting value for a full field from inference of one scalar; the
  initial field remains distinct from the prior mean.
- Existing optimization accepts finite conductivity bounds; the proposed
  log-Gaussian prior only enforces positivity. Q10 selected positivity alone for
  the first calculation.
- Forward and adjoint share conductivity and temperature objects. The exposed
  field after new inference calls is specified by Q9; Q13 settles numerical
  failure handling and interruption.
  The temperature associated with a mean conductivity is not generally the
  posterior mean temperature.
- The forward solver accepts known heat sources and bottom temperatures beyond
  the default example. HMC must inherit the supplied physical model rather than
  silently fixing the bottom temperature to 300 K.
- `notebooks/MultiFidelitySolve.ipynb`, cell 16 onward, recovers spatial ordering
  from coordinates and computes transfer and correlation spectra. Conductivity
  draws should retain an unambiguous relationship to the spatial mesh.
- Current XDMF export is a single-field snapshot, not a chain or restart file.
  Saving, continuation, warmup retention, and posterior-diagnostic dependencies
  are settled by Q13/Q14 and require new supporting calculations.
- A posterior mode depends on whether its density is expressed in conductivity
  or log conductivity. Q12 selects the latter explicitly for the first model.

## Decision tree

The first round contains independent decisions. Exact calling sequences and
output behavior follow only after the statistical relationship and compatibility
requirements are settled.

1. **Q1: relationship between optimization and sampling — accepted.**
   Should a single declared likelihood and prior support both a best-fit field
   and HMC, or may HMC describe a separate statistical model while the original
   reconstruction remains separate?
   Decision: a common posterior for the new optimization and sampling
   workflows; the original estimate may also supply an initial field.
   Dependent decisions: definition and coordinates of a posterior mode; shared
   observations and prior; construction of the inverse problem; initialization
   of sampling from an optimized field.
2. **Q2: compatibility with existing scientific calculations — accepted.**
   Must existing notebook calls retain their present meaning and numerical
   behavior, or is an explicit migration acceptable?
   Decision: retain existing calls and their meaning, with an explicit
   new workflow for the declared posterior.
   Dependent decisions: accepting existing observations and adjoint objects;
   migration of fixtures; notebook changes; common plotting and field ordering.
3. **Q3: first useful scientific workload — sequence accepted; see Q7.**
   Which inference mesh and computing arrangement must the first usable HMC
   calculation support? Is a single process per chain sufficient, or must one
   temperature solve already span multiple MPI ranks?
   Decision: validate on a small full-field mesh, then measure a nominated
   representative workload; do not equate a small example with production-scale
   convergence or leave the practical target unspecified.
   Dependent decisions: number of chains and resource limits; sampler tuning;
   numerical tolerances; saving and restarting; convergence and cost criteria.

## Round 2: newly available decisions

4. **Q4: notebook workflow — accepted.**
   Should one inverse-problem description, constructed from an existing forward
   model plus observations and prior, provide both optimization and sampling?
   Accepted workflow, with illustrative names:

   ```python
   problem = HeatInverseProblem(forward=fwd, observations=data, prior=prior)
   estimate = problem.optimize(initial_h=h0)
   draws = problem.sample(initial=estimate, seed=42)
   ```

   These are proposed names, not implemented interfaces. The user supplies
   conductivity in physical units; the consistent adjoint and sampling
   coordinates are managed by the calculation. This refines the earlier
   `initial_h=estimate.h` example: passing the complete estimate also preserves
   additional inferred fields if future priors introduce them. Direct
   initialization from a positive physical field remains available for the
   baseline prior with its invertible map. Q12 selects a posterior mode in log
   conductivity.
5. **Q5: observations and noise — accepted.**
   Decision: accept fixed sensor positions, temperatures, and known
   independent errors, with either a common standard deviation or one per
   sensor. Provide an explicit conversion of existing nodal observations using
   their original physical coordinates and ordering. Correlated or inferred
   errors would be a further physical requirement, not an automatic conversion.
6. **Q6: physical structure of conductivity — clarified and accepted.**
   The user eventually wants sharp variations in `h(x,y)` and requires the
   calculation to permit suitable priors later. The Gaussian baseline may
   smooth those variations and is not expected to recover the truth perfectly.
   Keep the physical temperature prediction and likelihood separate from the
   conductivity prior. A different prior may add parameters or change sampling
   coordinates/method; support for every future prior in the same Gaussian
   map is not implied.
7. **Q7: concrete initial workload — accepted, with Q11 study sequence.**
   Test all three meshes: `nmesh=16, 32, 64` (289, 1089, 4225 unknowns). Use
   the same physical observations when comparing inference resolutions. The
   initial reply specified resolutions; Q11 subsequently settled the study
   sequence and computing arrangement. No fixed chain length or runtime budget
   has been imposed.

## Round 3: workflow and physical interpretation

The user agreed to the recommendations for Q4 and Q8–Q11 together. Method names
in examples remain illustrative, and the agreed physical behavior is binding.

8. **Q8: fields returned for analysis — accepted.**
   Decision: expose physical conductivity draws with chain/draw/spatial
   ordering, coordinates, mean, standard deviation, and pointwise quantiles,
   together with sampling diagnostics and access to sensor predictions. Preserve
   complete sampling coordinates separately for continuation and future priors.
   Existing spatial plots should accept a selected physical draw or summary.
   Pointwise credible intervals must not be labelled simultaneous field bands.
9. **Q9: supplied forward model after inference — accepted.**
   The current solvers share mutable conductivity and temperature fields.
   Decision: the new optimization/sampling workflow restores the supplied
   forward model's pre-call physical state on return or handled interruption.
   Loading a selected draw or summary is explicit and recomputes its temperature,
   so the visible conductivity and temperature correspond. Posterior mean
   temperature is a separate quantity from temperature at posterior mean
   conductivity. Existing TAO calls retain their established behavior under Q2.
10. **Q10: prior scales and conductivity support — accepted.**
    Decision: for the first Gaussian calculation, explicitly specify and
    hold fixed the prior mean, correlation, and amplitude parameters, and use
    positivity as the only conductivity bound. Expose prior draws and run prior
    sensitivity checks; do not inherit the original reconstruction's numerical
    bounds or use its initial field as a prior mean automatically. Inferring
    those parameters, or imposing physical upper/lower limits, changes the
    posterior and is outside the first release.
11. **Q11: evidence at the three required meshes — accepted.**
    Decision: run numerical checks, then a smooth-truth case on all three meshes using four
    independently seeded chains, one process for each heat solve, and report
    diagnostics and cost. Run a sharp-variation stress case on the same meshes
    under the Gaussian baseline to assess bias and uncertainty. Exact recovery
    is not required. Choose run lengths after pilot timings and report any
    unconverged case explicitly. Q15 subsequently sets qualification thresholds
    and requires preserving flagged unconverged results; no runtime cap has been
    supplied. Parallel independent chains are distinct from distributed heat
    solves.

## Round 4: final scientific and run-behavior choices

The user agreed to all five recommendations below. They are accepted
requirements, and the specification now incorporates them.

12. **Q12: definition of the optimized posterior mode — accepted.**
    Decision: optimize the posterior density in log conductivity for the
    fixed Gaussian baseline and label that coordinate choice. Its affine map to
    `z` means the same mode is obtained by minimizing `U(z)`. Return the
    corresponding physical conductivity, but do not call it the mode of the
    physical-conductivity density: that density has negative log proportional
    to `U(z(m)) + sum(m)`. It is also distinct from posterior mean conductivity.
    For initialization accept a positive uniform field, a supported callable,
    or conductivity values associated with the declared mesh; a uniform start
    still represents full-field inference. Four study chains use distinct seeds
    and dispersed starting states, not identical deterministic trajectories.
13. **Q13: saving, continuation, and interrupted calculations — accepted.**
    Decision: save draws incrementally together with the declared
    physics, mesh, sensors/noise, prior, seeds, and numerical settings. At
    completed trajectory boundaries save the full state needed to continue,
    including the random generator and warmup state. Distinguish warmup from
    retained draws and check that a resumed run uses the same posterior.
    Continuation in the same recorded numerical environment should reproduce an
    uninterrupted run; changed numerics must not silently append to it. A
    handled interruption preserves the last complete saved state and restores
    the caller's forward model. An unconverged heat/adjoint solve stops with
    evidence; a nonfinite Hamiltonian proposal is rejected and counted without
    changing conductivity bounds or the target density.
    The saved chain includes rejection repeats, and its draw records and
    continuation state refer to the same completed trajectory.
14. **Q14: sampler controls in the notebook — accepted.**
    Decision: the normal call requests warmup, retained draws, and seed;
    warmup tunes the step size automatically, then adaptation stops. Expose
    optional explicit numerical controls for scientific comparisons. Use the
    exact Gaussian rotation with likelihood-force kicks and randomized
    trajectory durations as the initial HMC method; the returned settings and
    diagnostic report identify this method. The initial discussion allowed a
    native NumPy/SciPy implementation; Q17 subsequently selected BlackJAX with
    these same dynamics. Use an established optional diagnostics dependency
    rather than implement rank-normalized diagnostics afresh. General NUTS or
    other priors remain later extensions. Fixed PDE tolerances are qualified
    by separate accuracy checks and do not adapt along retained trajectories.
15. **Q15: when a posterior result qualifies for scientific summaries — accepted.**
    Decision: across four chains require rank-normalized split/folded
    `R-hat < 1.01`, combined bulk and tail ESS at least 400, and zero recorded
    post-warmup divergences for the initial demonstrations. Apply diagnostics
    to every nodal conductivity and the reported nonconstant physical summaries;
    record Monte Carlo errors, worst diagnostics, and where failures occur.
    Identify analytically fixed quantities separately; nonfinite diagnostics on
    inferred quantities do not pass by omission. These are practical diagnostic
    thresholds, not proof of global exploration or prior adequacy. Preserve and
    label unconverged results rather than suppress them or change the target to
    improve diagnostics. Numerical validity and sharp-truth model adequacy are
    assessed independently. Nonfinite potential, force, or Hamiltonian
    trajectories count as divergences; ordinary rejection does not. Guidance:
    [Stan diagnostics](https://mc-stan.org/learn-stan/diagnostics-warnings.html),
    [bulk and tail ESS](https://mc-stan.org/rstan/reference/Rhat.html),
    [Vehtari et al.](https://arxiv.org/abs/1903.08008).
16. **Q16: observation experiment for the validation studies — accepted.**
    Decision: the main 16/32/64 comparison uses one fixed 16-by-16
    interior sensor array (256 observations) and known noise `sigma=1e-3 K`,
    matching the current fixture's noise amplitude. Generate each truth case's
    sensor data once on a finer mesh and reuse them across inference meshes.
    Keep the original nodal fixture as a separate input-compatibility check at
    its original physical nodes. The alternative of using every fine-mesh
    nodal observation as the primary experiment was not selected; it would
    supply much more temperature information. Prior parameters remain explicit;
    sensitivity is assessed without tuning the prior to force recovery of the
    known truth.
    Hold the same physical prior across the main mesh comparison and identify
    prior-sensitivity studies separately.

This record covers Q1–Q16. The follow-on implementation interview records
Q17–Q24, all accepted by 2026-10-26. Review the consolidated specification as
the final confirmation of shared understanding before the implementation plan.
Routine module names, file layouts, and numerical choices are implementation
responsibilities governed by these requirements, rather than further user
questions.

## Documentation practice

Update `CONTEXT.md` when a physical term is agreed. Record substantial accepted
trade-offs in short entries under `docs/adr/` when they merit a durable decision
record. Update the proposal as decisions settle, and ask for confirmation of
shared understanding when the decision tree is exhausted.
