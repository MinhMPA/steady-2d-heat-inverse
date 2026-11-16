# Conductivity posterior interfaces

The physical inputs are a fixed heat experiment, thermometer readings and their
known errors, a full-field log-conductivity prior, and fixed solve tolerances.
The [tutorial](../hmc.md) gives a complete experiment; the
[theory](../theory.md#fixed-sensor-conductivity-posterior) derives its probability
law and adjoint. The [saved-result guide](../hmc-results.md) gives continuation,
array export and closure examples. Imports remain flat top-level modules.

## Physical inputs

`N` denotes conductivity nodes, `S` thermometers, `C` chains, `D` selected retained
iterations and `Q` requested probabilities. Physical saved fields use canonical
node order: row `i` of `coordinates` labels field entry `i`, and `cells` indexes
these rows. Keep coordinates and connectivity with exported values. Native
finite-element functions instead use their own function-space ordering.

| Object | Scientific meaning and accepted inputs |
| --- | --- |
| `TemperatureObservations(points, values, sigma)` | Immutable copies of `(S,2)` physical positions and `(S,)` measured temperatures in kelvin; `sigma` is a positive finite scalar or `(S,)` standard deviation in kelvin. No parameters have defaults. Sensor order is preserved. Noise amplitude is known, not inferred. |
| `TemperatureObservations.from_function(T_obs, *, sigma)` | Copy all original scalar P1/Q1 nodal measurements and physical positions on one MPI rank. A vector `sigma` follows that function's original native order. |
| `GaussianLogConductivity(*, mu, kappa, tau, h_ref)` | No parameter defaults. `mu` is the dimensionless mean log field; `kappa`, `tau` and `h_ref` are finite positive scalars. Their units and elliptic covariance are defined in the theory. `mu` accepts a scalar, callable, coordinate-labelled `NodalField` or scalar P1/Q1 function. Binding copies its evaluated values. |
| `HeatNumerics(rtol=1e-10, atol=1e-12, max_it=10000, ksp_type="cg", pc_type="hypre")` | Fixed relative/absolute linear-solve tolerances, positive iteration cap, and PETSc method/preconditioner names. These apply to heat and adjoint calculations. They are numerical controls, rather than measurement errors or a prior. Their posterior accuracy still needs experiment-specific qualification. |
| `NodalField(values, coordinates)` | A field with `(N,)` real finite values and matching `(N,2)` physical labels. Conversion into the experiment validates and reorders those labels. Use this for arrays rather than assuming an arbitrary nodal numbering. |

Physical conductivity inputs to `initial_state` and `optimize` accept a scalar,
a callable with the forward solver's coordinate convention, a scalar P1/Q1
finite-element function, or `NodalField`. Bare arrays are refused because they
lack physical labels. Conductivity must be finite and strictly positive; a
log-field mean `mu` may be negative. Mismatched, ambiguous or missing coordinates
are rejected. The prior has natural Neumann boundaries on every wall and keeps
every conductivity node, including those on the imposed-temperature wall.

`SensorOperator(V, points)` predicts `(S,)` thermometer values from `(N,)`
**native** temperature coefficients via `predict(T_dofs)`. Its
`transpose(weights)` returns `(N,)` dual temperature coefficients directly from
`(S,)` weights, with no likelihood mass weighting. Sensors outside the mesh are
rejected; no physical extrapolation is introduced.

```{eval-rst}
.. automodule:: temperature_observations
   :members: TemperatureObservations, SensorOperator

.. automodule:: conductivity_prior
   :members: GaussianLogConductivity, BoundGaussianPrior

.. automodule:: heat_model
   :members: HeatNumerics, NodalField
```

## Declared experiment, state and optimized estimate

`HeatInverseProblem(*, forward, observations, prior, numerics=None)` freezes the
caller's mesh/connectivity, source and imposed temperatures, plus independent
copies of the observations and evaluated prior. `numerics=None` selects
`HeatNumerics()`. A matching bound Gaussian prior is also accepted. The current
native inference calculation requires one MPI rank and scalar P1/Q1 fields.
The supplied forward solver remains caller-owned: constructing the problem,
optimizing and sampling preserve its current conductivity and temperature.

`problem_id` labels the physical experiment, noise law and prior;
`numerics_id` separately labels recorded numerical settings. Neither includes
the caller's present trial conductivity or temperature. The known omitted
form-compiler settings mean these identities alone cannot yet certify unchanged
numerics; see the [current limitation](../validation/hmc-status.md#numerical-reproducibility).

| Call or result | Meaning |
| --- | --- |
| `initial_state(value)` | Return a copied complete `InferredState` from a physical field, a same-experiment state, or a `PosteriorEstimate`. This Gaussian field can invert physical conductivity to its complete coordinates. Existing complete states need no inversion. No heat solve is required. |
| `InferredState(coordinates, problem_id)` | A complete coordinate dictionary and its physical-experiment identity. Here the dictionary is `{"z": z}` with shape `(N,)` in canonical physical node order. Arrays are copied; the problem checks identity, finite values and the complete prior-declared shape when accepting it. |
| `optimize(*, initial_h, maxiter=500, gtol=1e-6, ftol=1e-12)` | Use the heat/adjoint energy and force to minimize this posterior over every whitened coordinate. `maxiter` is positive; the gradient and relative-energy stopping tolerances are finite and nonnegative. Returns `PosteriorEstimate`. |
| `PosteriorEstimate(state, conductivity, objective, success, message, mode_coordinates="log_conductivity")` | `state` is complete; `conductivity` is a copied `(N,)` positive physical field; `objective` is `U(z)`. `success` and `message` retain actual termination evidence. Its mode is in log conductivity, not the density in physical conductivity and not the physical posterior mean. |

Inspect `estimate.success` and `estimate.message` before calling a candidate a
converged optimum. A nonfinite trial after a finite start returns the last finite
candidate with `success=False`; a nonfinite initial evaluation cannot return a
preceding candidate. An unconverged heat/adjoint solve raises `HeatSolveFailure`.
No legacy conductivity bounds or conductivity-density Jacobian are added.
The original TAO reconstruction remains a different objective.

## Sampling and continuation

```text
sample(*, initial=None, seed, warmup=1000, draws=2000, chains=4,
       output=None, controls=None, initial_states=None, progress=None)
resume(output, *, total_draws=None, progress=None)
```

These are `HeatInverseProblem` methods. `sample` returns an open
`PosteriorSamples` handle after the requested calculation completes. Counts are
**per chain**: defaults are four independent chains, 1,000 warmup trajectories
and 2,000 retained trajectories each. `seed` is a required unsigned 32-bit
master seed. Supply exactly one common `initial` or `initial_states` with one
complete accepted state per chain. Multiple chains around a common center start
at `z_center + initial_spread * independent_standard_normal`; a single chain and
explicit per-chain states use their supplied starts exactly.

`controls=None` selects the conditional defaults below. `output` must be a new
calculation directory; `None` creates a unique directory under `runs`.
`progress`, if supplied, is called after committed trajectory boundaries with
per-chain progress evidence. It does not change the posterior.

`HeatInverseProblem.load(output)` reconstructs a saved physical experiment and
its private workspaces. `resume` finishes its saved target; `total_draws=N`
requests **N retained states in total per chain**, at least the original target,
with a new matching diagnostic report required after extension. Existing warmup,
fields, starts, random states and controls persist. Changed recorded identities
are refused before append, subject to the form-compiler limitation above.
A failure propagates with the saved output path; committed history remains
inspectable. Neither operation starts a new physical experiment implicitly.

Use problem and result context managers, or call `close()` explicitly; repeated
closure is harmless. Closing the problem releases its owned heat/adjoint
workspaces and preserves the caller-owned forward solver. Closing a result
releases its archive handle. Until the reviewed chain-resource defect is fixed,
a completed chain may retain an adjoint field/form until the enclosing problem
closes; result closure alone does not release those objects.

```{eval-rst}
.. automodule:: heat_inference
   :members: HeatInverseProblem, InferredState, PosteriorEstimate
```

## Saved physical fields and summaries

`PosteriorSamples.open(output)` opens a complete or interrupted archive. Reading
its physical arrays, `metadata`, `coordinates` and `cells` needs only NumPy and
the standard library, without a new heat solve or MPI/JAX/BlackJAX. `output` is
the resolved archive directory; metadata is copied and spatial labels are
immutable. Chain and retained-draw indices start at zero; `stop` is exclusive.

| Method | Returned physical quantity and axes |
| --- | --- |
| `field(*, chain, draw)` | Copied physical conductivity `(N,)`. |
| `load_draws(*, chains=None, start=0, stop=None)` | Explicit allocation `(C,D,N)` in selected chain order; default is every chain and all retained states. Unequal lengths require an explicit common range or equal-length selection, rather than silent truncation. |
| `sensor_predictions(*, chain, start=0, stop=None)` | Saved latent thermometer temperatures `(D,S)` in original sensor order, without a solve; future readings also contain measurement noise. |
| `state(*, chain, draw)` | Complete `InferredState`; lazily imports the native inference classes. |
| `summary(*, quantiles=(0.05,0.5,0.95), node_chunk_size=64)` | `FieldSummary` over every retained state, in bounded node chunks. It does not run diagnostics. |
| `temperature(*, chain, draw)` | One private heat solve returning physical temperatures `(N,)` in canonical saved order. Requires native finite-element/MPI imports. |
| `as_function(*, chain=None, draw=None, summary=None)` | A private conductivity function and mesh, without a heat solve. Select one complete draw or `"mean"`, `"sd"`, `("quantile", p)`. Requires native imports; its coefficients use its own function-space order. |
| `apply_to(forward, *, chain=None, draw=None, summary=None)` | Explicitly apply one draw, mean or physical quantile and solve the destination temperature, returning that destination's temperature function. The destination's mesh, source and boundaries must match; a mismatch or failed application preserves it. `"sd"` cannot be applied as conductivity. |

`FieldSummary.mean` and `.sd` have shape `(N,)`; `.sd` is the population standard
deviation of physical draws. `.quantiles` has shape `(Q,N)` in `.probabilities`
order; `.coordinates` is `(N,2)`, `.cells` is connectivity and `.counts` is `(C,)`.
These arrays are immutable. Warmup is excluded, but every retained rejection
repeat contributes. Unequal partial chains contribute in proportion to their
available counts and cannot qualify an unfinished calculation. With no retained
draws, `summary` raises `ValueError`.

Default quantiles are 5%, 50% and 95%. To obtain the displayed equal-tailed
pointwise **68% and 95%** intervals request
`(0.025, 0.16, 0.5, 0.84, 0.975)` explicitly. Transform before averaging:
`mean` is $E[h]$, and applying it computes $T(E[h])$, generally different from
$E[T(h)]$. A pointwise interval is not a simultaneous field envelope.

```{eval-rst}
.. automodule:: posterior_samples
   :members: PosteriorSamples, FieldSummary
```

## Diagnostic evidence

`diagnostics(*, node_chunk_size=64, quantiles=(0.05,0.5,0.95))` reads saved
conductivity and thermometer histories and returns `DiagnosticReport`. It
optionally imports the diagnostic extra and attempts to save `diagnostics.json`;
it needs neither JAX nor another heat solve. Use the same probabilities for
`diagnostics` and `summary`. `.qualified` on a summary requires a current,
matching, qualifying saved report; missing, stale, incomplete or unavailable
reports leave it false. Probabilities 0 or 1 cannot qualify quantile errors.

The report's `.status`, `.qualified`, `.reasons` and `.persisted` distinguish
scientific qualification, missing evidence and whether the advisory report was
saved. `.counts`, `.common_draws` and `.requested_total` describe actual and
requested per-chain lengths; diagnostics inspect the common prefix.
`.per_quantity` maps physical quantities to statistics with a final node/sensor
axis; quantile Monte Carlo errors have `(Q, location)` axes in `.probabilities`
order. `.worst_locations` locates spatial weaknesses. `.divergences` counts
retained divergences; `.identity`, `.settings` and `.package_version` label the
evidence. Unavailable diagnostic packages yield an explicit unavailable,
unqualified report rather than silently passing.

Qualification requires four completed chains, rank-normalized split/folded
R-hat < 1.01, bulk ESS >= 400, tail ESS >= 400 and no retained divergences for
monitored nonconstant quantities, with finite required Monte Carlo errors.
Analytically fixed boundary thermometers have undefined R-hat/ESS and zero
Monte Carlo errors. Their separate `.fixed_sensor_consistency` records raw
residuals against prescribed temperatures; every nonzero discrepancy remains
numerically unverified. Convergence alone establishes neither calibration nor
continuum accuracy. See the [dated evidence](../validation/hmc-status.md).

```{eval-rst}
.. automodule:: posterior_diagnostics
   :members: DiagnosticReport
```

## Trajectory controls

`HMCControls` is frozen and records the following defaults. All times and
spacings are dimensionless Hamiltonian time in whitened coordinates.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `initial_step_size` | `None`, resolved to `6.5e-4` | Initial integration spacing; `initial_step_size_explicit` records whether it was supplied. |
| `target_acceptance` | `0.8` | Desired Metropolis acceptance for dual averaging. |
| `trajectory_interval` | `(pi/4, 3*pi/4)` | Randomized duration interval, preserved during sampling. |
| `adapt_step_size` | `True` | Adapt spacing during warmup only; momentum covariance stays identity. |
| `step_size_bounds` | `(1e-6, 1.3e-3)` | Allowed spacing; an adaptation demand below the lower bound raises. |
| `divergence_threshold` | `1000.0` | Absolute Hamiltonian energy-error threshold. |
| `initial_spread` | `1.0` | Whitened Gaussian spread for multiple chains around a common center. |

These settings are informed by selected curvature and three-mesh pilots, with
a mesh-16 comparison of spreads `0.1`, `0.3`, `1.0`; they require matching
qualification for each prior, data and mesh. They do not establish universal
trajectory stability or production convergence. The tutorial separates its
short illustrative controls from these defaults. Zero warmup requires
`adapt_step_size=False` **and an explicitly supplied spacing**, even if its
value is `6.5e-4`.

Controls reject nonfinite scalars, require `0 < Tmin < Tmax < pi`,
`0 < lower <= upper <= min(Tmin/2, pi-Tmax)`, and an initial spacing within
bounds. Target acceptance lies strictly between zero and one, divergence
threshold is positive and spread is nonnegative.

`require_sampling_environment()` checks CPU float64, Python and optional-package
versions; the physical inference setup separately enforces one MPI rank. The qualified combination is
CPython 3.12, BlackJAX 1.7.1, JAX/JAXlib 0.11.2; diagnostic qualification uses
ArviZ Stats 1.3.3. Sampling imports are deferred until needed. Optimization needs
the native heat/adjoint stack and SciPy, without JAX/BlackJAX. See
[installation](../installation.md) for distinct optional dependencies.

Invalid input types/shapes, inconsistent identities, invalid selections and
controls raise rather than silently changing the experiment. Numerical
nonconvergence raises `HeatSolveFailure`; nonfinite trajectory arithmetic is a
divergent rejection. The [theory](../theory.md#split-hamiltonian-dynamics)
describes the exact rotation, approximate likelihood kicks, Metropolis correction
and failure distinctions. Source details below are reference material, not
additional setup steps for a first experiment.

```{eval-rst}
.. automodule:: hmc_blackjax
   :members: HMCControls, require_sampling_environment
```
