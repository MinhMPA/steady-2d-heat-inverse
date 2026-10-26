# HMC implementation and validation interview

Opened: 2026-10-18. Updated: 2026-10-26. Status: Q17 selected BlackJAX with
the agreed split HMC; Q18–Q24 are accepted. All identified requirements are
settled and incorporated into the consolidated specification for final review.
This continues the accepted Q1–Q16 requirements. Library selection does not
establish installation, numerical qualification, or completed implementation.

The user invoked `improve-codebase-architecture` and `grill-with-docs` to examine
the notebook interface, supporting calculations, numerical validation, and use
of external JAX sampling libraries with the existing heat model. Discuss these
from the conductivity-inference viewpoint first. The architecture report lives
in the OS temporary directory, as required by the invoked skill.

## Starting point

- Branch `feature/hmc-sampling`; reviewed revision `149f3a0`.
- The [scientific specification](2026-10-16-full-field-hmc-design.md) and
  [Q1–Q16 record](2026-10-17-hmc-requirements-interview.md) define the posterior,
  physical outputs, state restoration, continuation, and validation studies.
- ADRs 0001–0003 remain in force: one posterior for new optimization and
  sampling, replaceable conductivity priors, and preservation of the supplied
  physical state.
- Q14 selected exact Gaussian-prior rotation with likelihood kicks, identity
  momentum covariance, automatic step-size tuning, and randomized trajectory
  durations. It allowed the initial NumPy/SciPy implementation and deferred
  BlackJAX/NUTS. Q17 now selects BlackJAX for the first implementation while
  preserving those dynamics. Standard NUTS remains a possible later comparison.
- No HMC production code exists. The `s2dhi-011` inspection on 2026-10-18 found
  DOLFINx 0.11.0, PETSc 3.25.4, NumPy 2.5.2, SciPy 1.18.0, and h5py 3.16.0.
  Metadata inspection then found no JAX, jaxlib, BlackJAX, NumPyro, ArviZ, or
  ArviZ Stats installation. No package was installed during this review; verify
  the environment again before numerical qualification.

## Facts that constrain the implementation

The current adjoint shares conductivity and temperature with the forward model
(`src/adjoint_solver.py:71–73`), and its source uses an integrated temperature
residual (`:99`). The optimizer also evaluates an integrated residual and
conductivity roughness (`src/tao_solver.py:252–263`). The new fixed-sensor
likelihood therefore needs a consistent sensor adjoint; connecting the existing
objective unchanged to a sampler would target a different probability law.

A scalar input becomes `fem.Constant`, including a scalar-returning callable
(`src/domain_coefficient.py:70–89`). The current optimizer correctly refuses to
promote that coefficient after forms are built (`src/tao_solver.py:83–91`). The
new calculation must make a uniform initial guess into a full conductivity
field before constructing its forms. Each chain also needs a deterministic,
controlled heat/adjoint calculation that satisfies the accepted state-restoration
and fixed-accuracy requirements.

The older Tier 3 plan assumes a different prior/Jacobian convention and changes
the legacy optimization route. Its `include_jacobian=True` recommendation must
not be copied into the accepted prior on log conductivity. The current target
is `U(z) = Phi(h(z)) + 0.5*z.T@z`; its likelihood force remains distinct from
the total posterior force.

Primary library documentation establishes possibilities, not a tested local
integration:

- [BlackJAX's non-JAX example](https://blackjax-devs.github.io/blackjax/examples/howto_other_frameworks.html)
  uses a host callback and a supplied derivative. The finite-element heat
  calculation need not be rewritten in JAX to explore this route.
- [BlackJAX HMC](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/hmc/index.html)
  accepts a custom integrator. Keeping the agreed Gaussian rotation requires
  explicitly selecting or implementing compatible dynamics, not merely calling
  the default HMC routine.
- [Window adaptation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
  tunes the mass matrix as well as step size. The accepted unit-frequency
  rotation requires identity momentum covariance; the separate
  [step-size adaptation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/step_size/index.html)
  machinery is a candidate for warmup without changing that momentum law.
- Both [BlackJAX NUTS](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
  and [NumPyro NUTS](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.NUTS)
  are available. NumPyro can receive an already-defined potential; choosing it
  does not require re-expressing the posterior as a second probabilistic model.
  Standard NUTS changes the agreed trajectory method. Library choice and
  trajectory method are separate decisions.
- [JAX external callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
  execute host code and assume purity for `pure_callback`; execution may be
  repeated or omitted. Exceptions from a pure callback are not a dependable
  failure mechanism. The implementation must demonstrate that a failed PDE
  solve stops the calculation rather than becoming an ordinary rejected draw.
- [Custom reverse derivatives](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html)
  can supply the adjoint without differentiating the finite-element code.
  This does not establish support for forward-mode or arbitrary higher
  derivatives. A JAX connection also does not move the PETSc heat solve to a GPU.

The exact package versions and numerical compatibility remain unqualified.
Current online documentation must not be treated as evidence that an arbitrary
installed release has every described feature.

## BlackJAX versus NumPyro — comparison requested 2026-10-22

This comparison informed the subsequent Q17 decision below. It did not itself
constitute acceptance of a library or a change to Q14. No sampler was installed
or numerically qualified for this comparison.

Both libraries can receive the same full-field target
`U(z) = Phi(h(z)) + 0.5*z.T@z`. BlackJAX receives its negative as a log density;
NumPyro can receive `U` directly through `potential_fn`. The latter route does
not require a second probabilistic model declaration. The conductivity map,
fixed-sensor likelihood, and prior remain defined once by this framework.

| Scientific task | BlackJAX | NumPyro |
| --- | --- | --- |
| Advance the field along a Markov chain | Explicit `init(position)` and `step(rng_key, state)`; transition information is returned separately. | `HMC` or `NUTS` supplies transitions; `MCMC` supplies warmup, repeated sampling, and collection. Lower-level transitions are also available. |
| Preserve the agreed Gaussian-prior dynamics | HMC exposes a custom integrator argument. We must supply and verify the Gaussian rotation and likelihood kicks. | Ordinary HMC/NUTS constructors use velocity Verlet and expose no integrator argument. A public `MCMCKernel` extension can supply custom dynamics to the existing driver. |
| Tune the step size with fixed identity momentum covariance | Separate dual-averaging machinery is available. Standard window adaptation also adapts the mass and must not be adopted unchanged. | Ordinary HMC/NUTS has independent step-size and mass-adaptation switches. A custom kernel must arrange its own appropriate adaptation. |
| Retain chain identity and diagnostics | Our driver collects returned states and transition information. | `get_samples(group_by_chain=True)` and `get_extra_fields(group_by_chain=True)` collect the chain and draw axes. |
| Continue a calculation | Persist the chain state, next random key, tuning state, trajectory controls, and calculation metadata. | `last_state` can become `post_warmup_state` for further draws using its saved random key. Durable saves and recovery during interrupted warmup still need project-specific work. |

Sources: [BlackJAX HMC](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/hmc/index.html),
[step-size adaptation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/step_size/index.html),
[NumPyro MCMC](https://num.pyro.ai/en/stable/mcmc.html),
[NumPyro HMC implementation](https://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc.html),
and [custom-kernel interface](https://num.pyro.ai/en/stable/_modules/numpyro/infer/mcmc.html).

For the custom rotation, the force distinction is essential: BlackJAX caches
`grad(logdensity) = -z - grad(Phi)`. The likelihood kick requires `grad(Phi)`;
using the cached total force directly would count the Gaussian restoring force
again. Momentum refreshment, full-energy acceptance, and rejected-state handling
can be reused, subject to numerical checks of our supplied integrator and
failure handling. Randomized durations remain part of the agreed experiment.

The finite-element connection is common to both choices. A JAX-facing energy
with a supplied reverse derivative can call the existing heat solve and a
consistent sensor adjoint. It does not move PETSc to a GPU or make the existing
integrated-residual adjoint into the sensor adjoint. Callback determinism,
independent chain-owned workspaces, and explicit solve-failure propagation need
qualification; throwing an exception inside `pure_callback` is not a reliable
implementation of the accepted failure behavior. See the
[external-model example](https://blackjax-devs.github.io/blackjax/examples/howto_other_frameworks.html)
and [JAX callback contract](https://docs.jax.dev/en/latest/external-callbacks.html).

The agreed notebook pattern can remain `problem.optimize(...)` followed by
`problem.sample(initial=estimate, ...)` under either library. Both routes still
need physical conductivity draws, sensor predictions, field coordinates,
restoration of the supplied heat model, and saved calculation provenance.
NumPyro's direct-potential route does not infer these physical meanings from
the array of Gaussian coordinates. Its probabilistic-model route may be useful
for future inferred noise or prior hyperparameters, but that is a future
scientific extension, not a prerequisite for using its sampler today.

Recommendation remains BlackJAX for the currently agreed split dynamics;
NumPyro is attractive if standard HMC/NUTS and its integrated run-and-collect
workflow become the priority. This is a fit assessment, not a speed claim.
Performance must be measured by effective physical-field draws per wall time
and per heat/adjoint solve on the agreed meshes. The qualification sequence
must also check energies and forces, known Gaussian targets, reversibility,
failure handling, and continuation before accepting either integration.

## Q17: which route should we implement first? — A accepted 2026-10-22

**A. BlackJAX with the agreed split HMC.** Retain the finite-element physics,
shared posterior, identity momentum law, Gaussian rotation, and likelihood
kicks. Reuse library transition and adaptation machinery through a small JAX
adapter. This advances external-library use into the first implementation.

**B. Standard NUTS through BlackJAX or NumPyro.** Prioritize a library method
that selects trajectory length automatically. This explicitly reopens Q14's
dynamics. Select the library after examining the consequences for the notebook
interface, failure handling, continuation, and cost per effective draw.

**C. Native split HMC first.** Preserve the earlier NumPy/SciPy route and defer
an external sampler. This simplifies the immediate connection to PETSc but
leaves more sampling logic and its maintenance within this project.

Decision: the user answered **A**. Use BlackJAX with the agreed split HMC for
the first implementation. Keep the shared posterior, Gaussian-prior rotation,
likelihood kicks, identity momentum law, automatic step-size-only warmup, and
randomized durations. This changes the earlier implementation sequence; it
does not change the target posterior or select standard NUTS.

Begin with a small numerical qualification of posterior values, adjoint forces,
Hamiltonian evolution, failed heat solves, and exact continuation before the
agreed four-chain studies at `nmesh=16, 32, 64`. A small native reference may
support verification; no second production sampler has been selected. Q18–Q24
below settle the subsequent scientific-use decisions. Routine implementation
details belong in the numerical plan. No new numerical result is claimed.

## Round after Q17: notebook behavior and independent calculations — accepted

The user answered **Agree** to Q18–Q21; acceptance is recorded on 2026-10-25.
These decisions refine Q4/Q8/Q12–Q14 and preserve the shared posterior, physical
outputs, automatic saving, and four-chain validation studies.

### Q18: interpretation of chain counts and run lengths

If a call requests four chains and 2,000 retained draws, should it return 2,000
draws per chain, with the warmup count also applying separately to each chain?

Accepted: expose the chain count directly, use per-chain counts,
exclude warmup from posterior summaries, and derive reproducible independent
chain random streams from one master seed. Exact study lengths still follow
pilot timings. Q22 subsequently selects four as the ordinary-call default.
Saving destination and qualified run-length defaults are implementation details.

### Q19: changing physical inputs after declaring a problem

After constructing the inference problem, the scientist changes the heat source
or supplied observation array. Should the existing problem retain its original
physical model and data, with changed experiments requiring a new problem?

Accepted: preserve fixed copies of the declared mesh, source, boundary
data, observations/noise, and prior when constructing the problem. An explicit
new problem declares a changed posterior. This does not make the caller's
current conductivity an implicit prior mean or starting field. Continue to
honor Q9's pre-call conductivity/temperature restoration. Each chain has its own
heat calculation using the fixed inputs. Q24 below resolves application to a
subsequently changed forward model.

### Q20: multiple starts from one optimized estimate

When a single complete estimate is supplied for four chains, should sampling
generate dispersed starts, or require four explicitly supplied inferred states?

Accepted: for multiple chains, allow the estimate to be the center of
reproducible prior-correlated perturbations, with a documented adjustable
dispersion, while also accepting explicit per-chain starts. For a single chain,
use its supplied starting state directly. Record actual starting fields and
the dispersion prescription; qualify sensitivity to that dispersion before
interpreting the four-chain diagnostics. Independent random seeds alone do not
make identical starting fields dispersed, and small perturbations near one
mode do not demonstrate exploration of other modes. A continuation restores
the saved states and does not repeat initialization. The precise dispersion
rule and pilot qualification remain to be specified.

### Q21: physical draws when a calculation exceeds memory

Should sampling return physical summaries plus access to saved draws on demand,
or load the whole physical chain into memory at the end?

Accepted: summaries and disk-backed access by default, with explicit
loading of all draws when wanted. A selected field should be available as an
ordinary array in the recorded spatial order for the existing plots. Retain
every post-warmup state, including rejection repeats, and compute summaries in
physical conductivity. At 4,225 unknowns, an illustrative four chains of 10,000
draws require 1.352 GB for one float64 field array alone; retaining the internal
coordinates adds another equally sized array. This is a storage illustration,
not a selected run length. File organization, exact-quantile computation, and
interruption/recovery mechanisms are implementation details governed by Q13
and Q23.

## Final identified round: ordinary use and continuation — accepted

The user answered **Agree with your suggestions** to Q22–Q24 on 2026-10-26.
These decisions complete the identified requirements questions.

### Q22: default number of chains

Should an ordinary `sample(...)` call default to four chains, with a single
chain available explicitly for exploratory checks?

Accepted: four by default, consistent with the intended scientific use.
A requested single-chain run remains available for quick numerical checks;
it cannot satisfy the already agreed four-chain qualification criteria. Four
chains alone do not establish convergence. Execution order or concurrency must
not change the meaning of a chain or silently reduce the requested count.

### Q23: finishing an interrupted calculation versus extending it

A calculation requesting 2,000 retained draws per chain stops at completed
counts `[2000, 2000, 1370, 0]`. Should resume finish the original request or add
another 2,000 draws to each chain?

Accepted: resume finishes the original target, adding only missing
iterations and completing any outstanding warmup. Requesting a longer run is
an explicit extension to a new per-chain total, for example 3,000. Completed
chains stay unchanged when merely resuming. Existing saved draws, random states,
tuning, physical inputs, and fixed numerical settings remain governed by Q13;
changing the target length alone does not repeat initialization or adaptation.

### Q24: applying a field after the physical model changes

Should ordinary application of a selected conductivity draw check that the
destination forward model still represents the recorded mesh, source, and
boundary conditions, or use its changed conditions automatically?

Accepted: check the physical inputs and report a mismatch before changing
the destination model. The comparison excludes its current conductivity and
temperature, which are the state being replaced. For a matching model, apply
the field and recompute its temperature as already required. Raw conductivity
fields remain available for
deliberately declared new experiments. Prediction under changed heating
conditions must be explicit and must not be labelled as a prediction for the
original experiment; a convenience operation for that new use case is not
required in the first release. No automatic transfer to a changed mesh is
included in the first release.

## Implementation responsibilities after this round

Q20 already specifies the scientific behavior of dispersed initialization. For
the selected Gaussian prior, the numerical plan can use
`z_initial[c] = z_estimate + s * xi[c]`, with independent standard-normal `xi`
and adjustable scalar `s`. Under the fixed linear map to log conductivity,
`s` scales the prior fluctuation amplitude. Compare several spreads in the
pilot and document a default based on the numerical and sampling evidence;
the agreement does not assign an untested universal numerical value. A small
spread near one estimate is not evidence of global posterior exploration.

The plan should retain sensor predictions needed for the accepted diagnostics
and compute selected full temperature fields when requested. Q8 requires
access to predictions; it does not require storing every full temperature
field. Exact physical quantiles can be computed in bounded-memory chunks from
the saved draws. Storage format, file organization, save destination defaults,
public argument names, package versions, and qualified numerical settings are
implementation choices to explain in the plan rather than additional user
preference questions.

The BlackJAX connection still needs the accepted checks of direct versus
connected energies/forces, deterministic heat calculations, split Hamiltonian
evolution, failed solves, and complete continuation. Tests must exercise the
same scientific calling sequence used by the examples. Selecting a library or
passing a short execution check does not satisfy these requirements.

## Scope carried into the numerical plan

Carry the following established responsibilities into the numerical plan:

1. What the scientist supplies and receives in the normal notebook sequence,
   and which controls need direct exposure for method comparisons.
2. Ownership of the heat calculation, snapshots of physical inputs, independent
   chains, and the supported process/precision arrangement.
3. What remains in the host posterior calculation and what the selected library
   owns; how value, force, and numerical failure evidence cross that connection.
4. Warmup, trajectory choices, interruption, saved state, and resumption under
   the selected library's actual behavior.
5. Evidence needed to adopt the route: direct-versus-mediated energies and
   forces, known targets and Hamiltonian checks, deliberate failure cases,
   exact continuation within the recorded environment, and measured solve cost.
6. How a failed qualification affects scope: no silent replacement of the chosen
   sampler or probability law.

The existing four-chain diagnostics and fixed-observation mesh studies remain
required. Favorable library diagnostics alone cannot validate the sensor
adjoint, discretization accuracy, or Gaussian prior adequacy.

## Review artifacts

The initial visual report presents the three considered routes with before/after
diagrams, their shared scientific responsibilities, and the necessary
qualification experiment. Its rendered layout was inspected. It is a historical
comparison; Q17's BlackJAX selection and the accepted written specification now
govern the work. Only documentation was changed during the interview; no new
numerical validation result is claimed.

All Q1–Q24 decisions are incorporated in the
[consolidated scientific specification](2026-10-16-full-field-hmc-design.md).
It is ready for final shared-understanding review. The remaining numerical
plan must make implementation choices concrete and testable within this scope.
