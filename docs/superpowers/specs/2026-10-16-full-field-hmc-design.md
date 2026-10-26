# Full-field Hamiltonian Monte Carlo for thermal conductivity

Status: requirements Q1–Q16 accepted on 2026-10-17; Q17 selected BlackJAX with
the agreed split HMC on 2026-10-22. Acceptance of Q18–Q21 is recorded on
2026-10-25; Q22–Q24 were accepted on 2026-10-26. All identified requirements
are settled. The user requested implementation planning on 2026-10-26; the
[implementation plan](../plans/2026-10-26-full-field-blackjax-hmc.md) is written
for review. HMC is not implemented yet. The
[requirements interview](2026-10-17-hmc-requirements-interview.md) records the
scientific requirements; the
[implementation interview](2026-10-18-hmc-implementation-interview.md) records
the library decision and subsequent accepted behavior.

The agreed first calculation infers the full conductivity field with a fixed
Gaussian prior on log conductivity. Optimization and HMC use one declared
posterior. The optimized estimate is explicitly a mode in log conductivity;
reported draws and field summaries are in physical conductivity. Sampling uses
exact Gaussian-prior rotations, likelihood-force kicks, automatic warmup, and
randomized trajectory durations. Long calculations save their progress and can
resume. Existing reconstruction calls preserve their scientific meaning.

The validation experiment has 256 fixed interior sensors with known noise
`sigma=1e-3 K`, held unchanged across `nmesh=16, 32, 64`. Four-chain diagnostics,
numerical accuracy, and prior adequacy are assessed separately. Sharp variations
are a future science case; the present Gaussian prior may smooth them, and exact
truth recovery is not required. Alternative priors discussed below are future
possibilities, not first-release deliverables.

Branch: `feature/hmc-sampling`, starting at
`0b452320b9468df1a685da49841885ed8c48f8c3`.

## Scientific objective

Infer a probability distribution for the complete spatial conductivity field from
noisy steady-state temperature measurements. Retain every conductivity degree of
freedom on the chosen finite-element mesh; do not truncate the field to a few
spatial modes. The user confirmed this choice on 2026-10-16.

The forward physics remains

\[
-\nabla\cdot(h\nabla T)=q,
\qquad T(x,0)=300\,\mathrm{K},
\]

with zero heat flux on the other walls. The heat source and temperature boundary
conditions are known. Uncertainty in them is outside this first calculation.
The value 300 K is the default example; the new inference calculation inherits
the heat source and boundary data of the supplied forward model.

## What the current calculation provides

The existing forward solver and discrete-adjoint gradient have an established
test suite. On the starting revision, the `s2dhi-011` environment passes 29 tests
and 4 subtests, including the existing gradient checks.

Baseline command (2026-10-16):

```bash
/Users/nguyenmn/miniconda3/envs/s2dhi-011/bin/python -m pytest -q --maxfail=1 --durations=5
```

The verified environment contains DOLFINx 0.11.0, PETSc 3.25.4, NumPy 2.5.2,
and SciPy 1.18.0. It is different from the Python environment initially active
in the shell. MPI initialization required execution outside the filesystem and
network sandbox, even for a single rank.

The current minimization objective is not the negative log posterior for its
synthetic observations: the misfit is an integral, equivalent to a finite-element
mass-matrix weighting, whereas noise is drawn independently at nodes. Also, the
roughness penalty on conductivity leaves its constant mode unconstrained.
Sampling requires an explicit likelihood and a proper prior.

The older Tier 3 plan already proposes measurements at fixed physical sensor
locations and one objective for optimization and sampling. The user confirmed
the shared-posterior requirement in interview Q1 on 2026-10-17 and the observation
inputs in Q5. Q10 fixed the prior parameters during each initial calculation;
their chosen values are recorded inputs, not universal defaults. This work does not declare
the older Tier 2 or Tier 3 plans complete.

## Initial prior and a considered alternative

1. **Selected baseline: Gaussian spatial prior on log conductivity.** Conductivity is
   positive, fluctuations are multiplicative, and both the spatial mean and the
   varying modes have finite prior variance. All modes are retained.
2. **Considered alternative: proper smoothness prior directly on conductivity.** Add a term
   constraining its mean to the existing roughness penalty, restrict to positive
   conductivity, and include the change-of-variables determinant when sampling
   logarithmic coordinates. This represents additive rather than multiplicative
   conductivity fluctuations and gives different credible intervals.

The selected prior is specified below. It is a physical modelling choice,
not an interpretation of the old regularization parameter `alpha`.

The Gaussian construction is the initial model, with other priors for sharp
variations to be supported later. The user explicitly does not require perfect
truth recovery in such cases; see
[Conductivity with sharp interfaces](#conductivity-with-sharp-interfaces) and
[`0002-allow-priors-for-sharp-conductivity.md`](../../adr/0002-allow-priors-for-sharp-conductivity.md).

## Measurements and likelihood

The data are temperatures `d` measured at fixed physical positions. For each
finite-element mesh, an interpolation operator `B` predicts the temperatures at
those same positions. The first noise model is independent Gaussian noise with
known, positive standard deviations, either one common `sigma` or an individual
`sigma_i` for each sensor (accepted in Q5):

\[
d=B T(h)+\epsilon,\qquad \epsilon\sim\mathcal N(0,\Sigma),
\qquad \Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_{n_s}^2),
\qquad
\Phi(h)=\tfrac12 [B T(h)-d]^T\Sigma^{-1}[B T(h)-d].
\]

For a common standard deviation this reduces to the original proposed
`Sigma = sigma**2 I`. Existing nodal temperature observations are accepted by
explicitly pairing them with their original physical coordinates and ordering.
They are not interpolated to new positions and then treated as having unchanged
independent errors. The same observation vector and uncertainties are used
when changing the inference mesh. Correlated errors and unknown noise
parameters are outside the accepted first observation model.

There is no mass matrix in this likelihood. The number and location of sensors
do not change when the mesh is refined. The interpolation supports the existing
linear triangular and bilinear quadrilateral elements. A fixed-temperature
boundary sensor can be represented, but contributes no conductivity information.

Q16 fixes the main example at a 16-by-16 interior sensor grid (256 temperature
readings), with common known noise `sigma=1e-3 K`. Generate each truth case's
observations once on a finer mesh and reuse the exact positions, data vector,
and uncertainties across inference meshes. Record the sensor coordinates,
truth mesh, truth field, and random seed before the comparison. Check the
reference temperature prediction under further refinement. The existing
nodal-noise fixture remains a separate compatibility check at its original
physical nodes.

## Positive conductivity and its spatial prior

Define dimensionless nodal variables

\[
m_i=\log(h_i/h_{\rm ref}),\qquad h_i=h_{\rm ref}\exp(m_i),
\]

where `h_ref > 0` has conductivity units. As in the present forward solver, the
finite-element conductivity interpolates these positive nodal conductivities.
This is distinct from exponentiating an interpolated log field inside a cell.

Use a Gaussian spatial prior constructed from the elliptic operator
`kappa**2 - Laplacian`, with positive `kappa` and `tau`. Let `M` be the
finite-element mass matrix, `K` the stiffness matrix, `M_L` the diagonal row-sum
mass matrix, and

\[
A=\kappa^2 M+K,\qquad
m=\mu+\tau^{-1}A^{-1}M_L^{1/2}z,
\qquad z\sim\mathcal N(0,I).
\]

The conductivity prior uses natural Neumann boundary conditions; the temperature
Dirichlet condition is not imposed on conductivity. Positive `kappa` constrains
the constant mode. The implied finite-dimensional covariance is

\[
C_m=\tau^{-2}A^{-1}M_L A^{-T}.
\]

This is a finite-element approximation to a Matérn-type Gaussian field in two
dimensions. `kappa` controls spatial correlation and `tau` controls amplitude.
Parameters are fixed during a sampling run and recorded with its output (Q10).
Their effects are explored through separate prior-sensitivity runs. Conductivity
is positive with no additional physical lower or upper bound in this first
model. An initial conductivity guess does not set the prior mean automatically.
The main 16/32/64 comparison holds `h_ref`, the prior mean field `mu(x,y)` on
the physical domain, `kappa`, `tau`, and prior boundary conditions fixed across
meshes; only their finite-element representation changes. Compare any different
prior parameters in separately identified sensitivity studies.
Neumann boundaries modify the covariance near the edges, so an infinite-domain
variance formula must not be described as the exact pointwise variance on this
square. Prior draws and refinement checks are part of validation.

HMC evolves the full vector `z`. This normalizes the prior variance of each
coordinate without discarding any degrees of freedom. The target potential is

\[
U(z)=\Phi(h(m(z)))+\tfrac12 z^T z.
\]

No conductivity Jacobian is added: the prior is defined in log conductivity and
represented by `z`. The linear map from `z` has a constant determinant for fixed
hyperparameters. If the alternative prior on `h` is chosen, the log-coordinate
potential instead requires the term `-sum(m)` (up to constants).

For a prior represented as a sufficiently differentiable map of independent
standard Gaussian coordinates, the likelihood force uses the map `z -> m` and
the action of its transpose derivative, `J(z)^T`. For the baseline Gaussian
prior, `J = tau**-1 A**-1 M_L**(1/2)` is constant; factorize `A` once and apply
the map through solves. These operations support that class of prior
representations, not arbitrary densities or discontinuous maps. A future prior
must explicitly supply its probability law and any derivatives required by its
sampling method; it must not silently inherit a Gaussian energy.

The baseline also has an inverse for initialization from positive conductivity:
`z = tau M_L**(-1/2) A (log(h/h_ref) - mu)`. Future hierarchical priors may
contain more unknown quantities than the conductivity field, so `h` alone need
not determine their sampling coordinates. The agreed optimization-to-sampling
handoff accounts for this distinction by accepting the complete optimized result
(Q4), while also accepting a physical initial field for the invertible baseline.

## Meaning of the optimized estimate

Q12 selects the mode of the posterior density expressed in log conductivity
`m`. For the fixed Gaussian baseline, the map between `m` and `z` is affine
with constant determinant, so minimizing `U(z)` finds the same mode. The result
returns its physical conductivity together with this coordinate interpretation
and the complete inferred state.

This field is neither the posterior mean conductivity nor generally the mode
of the density expressed in physical nodal conductivity. For this same
probability law, the latter density has negative log

\[
-\log p_h(h\mid d)=U(z(m))+\sum_i m_i+\text{constant}.
\]

The distinction is a change of density coordinates, not a change of physical
prior. A future nonlinear or hierarchical prior must identify what its optimized
density means; a joint mode need not be a marginal conductivity mode.

## Adjoint force and Hamiltonian dynamics

For residual `r = B T - d`, the temperature adjoint has source
`B.T @ (Sigma**-1 @ r)` and homogeneous temperature Dirichlet data. Its conductivity
derivative is assembled from `-grad(T) dot grad(lambda)`. Apply the nodal chain
rule and the transpose of the prior map to obtain

\[
\nabla_z U
=z+\nabla_z\Phi,
\qquad
\nabla_z\Phi=\tau^{-1}M_L^{1/2}A^{-T}
  \bigl[h\mathbin{\odot}\nabla_h\Phi\bigr].
\]

Each evaluation requires one temperature solve and one temperature-adjoint solve;
the prior operator is fixed and can be factorized once. No finite differences
are used to supply the sampling force.

The initial sampler is HMC with Gaussian momentum refreshment, identity momentum
covariance in `z`, and a Metropolis acceptance probability computed from the full
Hamiltonian

\[
H(z,p)=U(z)+\tfrac12 p^T p .
\]

The integrator treats the Gaussian part exactly instead of using plain leapfrog.
One step of size \(\varepsilon\) is

\[
p\leftarrow p-\tfrac{\varepsilon}{2}\nabla_z\Phi,
\qquad
\begin{pmatrix}z\\ p\end{pmatrix}\leftarrow
\begin{pmatrix}\cos\varepsilon & \sin\varepsilon\\ -\sin\varepsilon & \cos\varepsilon\end{pmatrix}
\begin{pmatrix}z\\ p\end{pmatrix},
\qquad
p\leftarrow p-\tfrac{\varepsilon}{2}\nabla_z\Phi .
\]

The rotation is the exact flow of \(\tfrac12 z^Tz+\tfrac12 p^Tp\), and each half
kick is the exact flow of \(\Phi\). The composition is reversible and
volume-preserving, so a Metropolis correction targets the declared finite
dimensional posterior, subject to numerical evaluation accuracy. Plain leapfrog
also approximates the Gaussian flow in each coordinate. The familiar
\(N^{-1/4}\) step-size scaling is established for an independent-product
asymptotic regime (Beskos et al. 2013), not as a universal exponent for this heat
posterior. Exact Gaussian rotation removes that source of energy error.
Hilbert-space results motivate robustness under refinement subject to regularity
and approximation assumptions (Beskos et al. 2011); those assumptions have not
been verified for this calculation. Validation item 7 measures acceptance and
cost rather than assuming either is independent of mesh resolution. Prior-only
integration (\(\Phi=0\)) preserves the Hamiltonian to round-off.

The kicks use the likelihood force \(\nabla_z\Phi\) only. Using \(\nabla_z U\),
which includes `z`, would count the prior twice.

The sampler draws a trajectory-time parameter `T` and takes
`max(1, round(T / epsilon))` steps; the actual duration is the step count times
`epsilon`. In these whitened coordinates, every
direction that the data do not constrain is an oscillator of unit frequency, so
a fixed `T` resonates: `T = 2*pi` returns all such directions to their starting
values, and `T = pi` reflects them. An exact rotation through `pi/2` gives an
independent proposal coordinate in an unconstrained direction. Draw `T` independently at each
iteration from an interval around `pi/2` that excludes `pi`, for example uniform
on `[pi/4, 3*pi/4]` (randomization as in Neal 2011). Randomized and rounded
durations do not give independent draws in general, and rejection of the whole
trajectory can repeat every coordinate. Check the actual durations when tuning
to avoid resonances. Q14 requires automatic step-size tuning during warmup and
fixed settings for retained draws. The normal calculation specifies warmup,
retained draws, and a seed; optional explicit controls permit numerical
comparisons. Record the realized step size, trajectory-duration distribution,
warmup settings, and integrator in the result.

Q17 selects BlackJAX for the first implementation, retaining the exact Gaussian
rotation, likelihood kicks, identity momentum covariance, step-size-only warmup,
and randomized trajectory durations. This supersedes the earlier allowance to
start with a native NumPy/SciPy sampler and defer BlackJAX. The heat equation
and consistent sensor adjoint remain finite-element calculations. A JAX-facing
potential with a supplied derivative connects that calculation to BlackJAX;
the connection and custom integrator must pass numerical qualification before
the four-chain mesh studies. Selection is not evidence of local compatibility
or completed implementation.

Standard NUTS remains a possible later method comparison. Its ordinary
leapfrog dynamics would again approximate the Gaussian flow, and its refinement
behavior would require separate measurement. It is not selected by Q17.
Use an established optional diagnostics package for rank-normalized statistics
and effective sample sizes, and record its version. The numerical implementation
plan chooses that package; its absence must be reported as diagnostics
unavailable, never as evidence of convergence.

Numerical requirements:

- Fixed, sufficiently tight tolerances for every temperature and adjoint solve;
  the optimization-specific adaptive tolerance controller is not used in HMC.
- Deterministic potential and force at a given `z`, independent of previous
  evaluations.
- Rejected trajectories retain the complete previous state and cached force.
- Nonfinite potential, force, or Hamiltonian trajectories are rejected,
  reported, and counted as divergences; no clipping of conductivity or silent
  alteration of the target density.
- An unconverged finite-element solve stops the run with a diagnostic rather
  than being silently interpreted as zero posterior probability.
- Each supported heat solve and its chain initially use one MPI rank.
  Independent chains have separate heat calculations. Distributing one heat
  solve across multiple ranks raises an explicit unsupported-operation error
  until its field distribution and global acceptance decision are separately
  validated.

## Conductivity with sharp interfaces

Accepted scope from Q6: the Gaussian model is the starting point, and the
calculation must permit other conductivity priors for future sharp-variation
studies. The heat equation and observation model remain reusable. A future
prior may add unknown fields or require different sampling coordinates or a
different sampling method; the first release need not implement every candidate
below. Exact recovery of sharp truth is not an acceptance criterion. Assess
bias, uncertainty, and predicted temperatures under the declared prior.

The baseline Gaussian prior is a Matérn-type field with smoothness
\(\nu=1\) in two dimensions; its draws are continuous. A true conductivity with
material interfaces conflicts with that continuity assumption. Inference can
smooth a transition and give biased estimates or inadequate uncertainty near
it. Knapik, van der Vaart & van Zanten (2011) establish coverage failures for
certain linear Gaussian inverse problems; this motivates a check here, not a
claim that the same outcome has been proved for nonlinear conductivity inference.

No prior removes the identifiability limits. Conductivity is weakly determined
where the temperature gradient is small. Interface localization depends on
temperature sensitivity, noise, sensor geometry, prior assumptions, and
discretization; sensor spacing alone is not a hard lower bound on localization
precision. A smooth posterior mean may also reflect uncertainty in the position
of an otherwise sharp interface. Report the inferred uncertainty and modelling
limitations rather than requiring a particular interval width.

The current conductivity uses continuous degree-one finite elements, so an
input jump is represented by transitions in cells crossing the interface.
Future exact discontinuities may require a different conductivity representation
as well as a different prior. The first calculation must state this distinction
when interpreting sharp-variation experiments.

The first calculation addresses the smooth case, such as the existing fixture
conductivity `1 + 6x^2 + x/(1 + 2y^2)`. Validation item 8 measures whether the
Gaussian prior is adequate for an interface at the chosen sensor density.
Candidate replacements for later scientific work are listed below. Their
implementation and selection are outside the first release:

1. **Two-layer Matérn prior** (Roininen, Girolami, Lasanen & Markkanen 2019).
   Let the correlation parameter vary in space, `kappa(x) = exp(ell(x))`, and
   give `ell` its own whitened Gaussian prior. All sampled coordinates remain
   whitened, so `U` is still `Phi` plus half the squared norm of the sampled
   vector, and no log-determinant term appears. With
   `u = A**-1 b`, `b = tau**-1 M_L**(1/2) z`, and
   `w = A**-1 (h * grad_h Phi)`, the force on the nodal value `ell_j` is
   \(-\int_\Omega (\partial\kappa^2/\partial\ell_j)\,u\,w\,\mathrm{d}x\), which
   equals \(-\int_\Omega 2\kappa^2\varphi_j\,u\,w\,\mathrm{d}x\) when
   \(\kappa^2=\exp(2\ell)\) is evaluated from the interpolated `ell` inside each
   cell; the derivative must follow whichever convention assembles `A(ell)`. Add
   `w_j b_j` if `b`
   is scaled nodally by `kappa` to control the nominal marginal variance.
   For constant coefficients on an infinite two-dimensional domain, the stated
   prior has variance `1 / (4*pi*tau**2*kappa**2)`: at fixed `tau`, shorter
   correlation means smaller variance. Spatially varying coefficients and
   boundaries require separate covariance checks; this amplitude scaling does
   not guarantee constant pointwise variance there.
   This force is then transformed by the transpose of `ell`'s prior map. The
   solve for `w` is already required by the force on `z`; the added cost is
   refactorizing `A(ell)` at every evaluation. The prior remains differentiable
   and compatible with HMC.
2. **Level-set prior** (Iglesias, Lu & Stuart 2016). The sign of a Gaussian field
   from the present construction selects one of two conductivity values, so
   draws are piecewise constant and the prior machinery is reused. The step
   function does not supply the smooth force assumed by this HMC proposal: use preconditioned
   Crank–Nicolson (Cotter, Roberts, Stuart & White 2013) in the same whitened
   coordinates, or smooth the step over a width tied to the mesh and accept the
   resulting bias.
3. **Cauchy difference priors** can favor large local changes. A proposed density
   on two-dimensional differences must be specified and normalized appropriately;
   it does not automatically come with the Gaussian representation used here.
   Direct sampling in `m` against its heavy-tailed density is one possible
   formulation and requires its own prior force and numerical validation.

## Deliverables

- A fixed-position sensor interpolation and Gaussian likelihood calculation.
- The proper log-conductivity prior and its full-dimensional normalized
  coordinates, implemented as the map from those coordinates to log
  conductivity together with the transpose of its Jacobian.
- A posterior potential and exact discrete-adjoint force.
- Optimization of that same declared posterior, so the new best-fit and HMC
  calculations share observations, noise assumptions, and prior. Label the
  estimate as the mode in log conductivity, as accepted in Q12.
- A reproducible HMC sampler, using the exact Gaussian splitting and randomized
  trajectory lengths, with warmup, recorded acceptance probabilities, energy
  errors, and divergence counts.
- A runnable example producing conductivity draws, posterior mean and standard
  deviation, and posterior predictions at sensors. Outputs include sensor data,
  mesh, prior parameters, numerical tolerances, seeds, and sampling settings.
- Incremental saving and resumable calculations, including complete inferred
  and random-generator states and separate warmup records.
- Documentation of the equations, example command, statistical diagnostics, and
  the distinction between implementation checks and a converged inference.

The new inference workflow must define the heat model, observations, noise model,
and prior once and support both posterior optimization and HMC. The existing
reconstruction calls must retain their original meaning and numerical behavior;
adopting the new posterior is explicit. These requirements were accepted in
interview Q1–Q2 and recorded in
[`0001-share-the-conductivity-posterior.md`](../../adr/0001-share-the-conductivity-posterior.md).

The new sensor adjoint can use the current forward solver by composition. This
does not require declaring the broad older refactor complete. New modules must
follow the repository's flat import convention and be registered in
`pyproject.toml`.

## Agreed scientific workflow and outputs

Q4 accepted the following calling pattern; names are illustrative until the
implementation plan fixes them:

```python
problem = HeatInverseProblem(forward=fwd, observations=data, prior=prior)
estimate = problem.optimize(initial_h=h0)
draws = problem.sample(initial=estimate, seed=42)
```

Q19 fixes the declared physical inputs when constructing the inference problem:
the mesh, source, boundary data, observations, noise model, and prior. Later
changes to the supplied forward model or original observation arrays do not
change that problem. A changed experiment is declared through a new problem.
Each chain has its own heat calculation using the fixed inputs. The supplied
conductivity remains a starting-state choice, distinct from the prior mean;
Q9's preservation of the caller's pre-call conductivity and temperature still
applies. Q24 governs application to a subsequently changed forward model below.

Q18 specifies warmup and retained-draw counts per chain. Four chains with 2,000
retained draws mean 8,000 draws in total, with the chain axis preserved. One
master seed determines reproducible independent chain random streams. Warmup
does not enter posterior summaries. Q22 sets four chains as the ordinary-call
default; an explicit single-chain calculation remains available for quick
checks. It cannot satisfy the four-chain qualification criteria, and four
chains alone do not establish convergence. Execution order or concurrency must
not silently reduce the requested chain count.

The user supplies conductivity in physical units. The inference calculation
constructs its consistent adjoint and sampling coordinates. A complete estimate
carries all inferred quantities as well as physical conductivity, so handing it
to sampling does not lose future hierarchical-prior parameters. The Gaussian
baseline also accepts a positive physical initial conductivity directly.
A uniform scalar, a supported callable, or mesh-associated conductivity values
can supply that starting field. A scalar initializes every conductivity unknown;
it does not reduce the inference to one scalar parameter. In particular, the
current forward solver's constant-coefficient representation must not silently
remove the field degrees of freedom. Four-chain studies use independent seeds
and dispersed starting states. Q20 accepts automatic dispersion for multiple
chains initialized from a single estimate: treat it as the center of
reproducible prior-correlated perturbations with a documented adjustable
amplitude, and also accept explicit per-chain starting states. Record the actual
starting fields and test sensitivity to the dispersion. A single chain uses
its supplied state directly. Continuation restores saved states rather than
repeating this initialization. The precise default dispersion is subject to
pilot qualification.

Q8 requires physical conductivity draws indexed by chain, retained draw, and
spatial degree of freedom, paired with mesh/coordinate information and the
ordering needed by existing plots. Return posterior mean, standard deviation,
pointwise quantiles, sampling diagnostics, and access to predicted sensor
temperatures. Retain internal inferred states for continuation. Compute means
and quantiles in the physical quantity being reported; distinguish pointwise
intervals from simultaneous field bands and temperature at mean conductivity
from mean predicted temperature.
Provide a direct route from a selected physical draw or summary to the existing
field plots. Access to sensor predictions must preserve the supplied physical
model just as optimization and sampling do. Numerical module names and storage
format are implementation choices; these physical meanings and behaviors are
the agreed requirements. Q21 selects physical summaries and access to saved
draws on demand as the default result. Loading the whole chain into memory is
explicit. A selected conductivity field is available as an ordinary array in
the recorded spatial order for existing plots. Summaries use all retained
states, including rejection repeats; memory limits do not justify silent
thinning or substituting summaries of logarithmic conductivity.

Q9 requires restoring the supplied forward model's pre-call physical state on
return or handled interruption. Applying a selected draw or summary is explicit
and recomputes its temperature. Q24 requires this ordinary application to check
that the destination mesh, heat source, and boundary conditions match the
recorded experiment before changing the destination model. A mismatch is
reported without applying the field. This comparison excludes the current
conductivity and temperature, which are the state being replaced. Conductivity
arrays remain available for deliberately declared new experiments; predictions under changed heating
conditions must be identified as such. A convenience operation for that use
case and automatic transfer to a changed mesh are outside the first release.
Original TAO calls retain their existing behavior. See
[`0003-preserve-the-supplied-physical-state.md`](../../adr/0003-preserve-the-supplied-physical-state.md).

## Saved calculations and continuation

Q13 requires automatic incremental saving. Each run records the physical model,
mesh and conductivity ordering, observations and errors, prior and reference
units, initial states, seeds, numerical tolerances, sampling controls, and
relevant software versions. Record enough to reconstruct the declared finite
dimensional posterior, not merely a callable's name. The saved result identifies
its location; unrelated previous calculations must not be overwritten.

Save completed draws and a recoverable continuation state at completed
trajectory boundaries. That state includes the full inferred coordinates,
current potential and force or the information needed to reproduce them,
random-generator state, warmup/adaptation state, and completed-iteration counts.
Keep warmup separate from retained draws. An interrupted write must leave a
previous complete saved state recoverable.
Record every retained iteration, including the repeated state after a rejected
proposal; accepted proposals alone are not posterior draws with the correct
weights. Draw records, random-generator/adaptation states, completed-iteration
counts, and diagnostic counters must correspond to the same completed
trajectory. Continuation must neither duplicate nor skip a retained iteration.

Resuming in the same recorded numerical environment must reproduce an
uninterrupted calculation, including interruptions during warmup. Validate that
the physics, observations, prior, mesh, and numerical settings still describe
the same calculation. Changed numerics cannot silently append to an existing
chain. A handled interruption restores the caller's physical model and leaves
the last completed trajectory available for continuation.

Q23 distinguishes resuming from extending. Resume completes the original
requested per-chain target, preserving chains that have already reached it.
For a 2,000-draw target interrupted at `[2000, 2000, 1370, 0]`, it completes only
the missing iterations and any outstanding warmup. Extension explicitly sets
a new per-chain total, for example 3,000. Neither operation repeats completed
initialization or warmup. Changing only the requested total does not authorize
changes to the posterior, chain identities, numerical settings, or saved
random states.

An unconverged heat or adjoint solve stops the calculation and preserves its
failure evidence and last complete saved state. A proposal with nonfinite
potential, force, or Hamiltonian is rejected and counted as a divergence, with
the previous valid state retained. Neither case permits clipping conductivity
or changing the target distribution. Define and record the additional finite
energy-error rule used to identify divergences before retained sampling;
ordinary Metropolis rejection is not itself a divergence.

## Validation before claiming a working sampler

Write each numerical check before the corresponding new implementation.

1. Sensor interpolation reproduces constant and linear fields, satisfies its
   transpose identity, and predicts the same physical sensors on different
   meshes.
2. The prior map, inverse, and transpose agree with an explicitly assembled
   covariance on a small mesh. Its constant mode has finite variance.
3. The complete posterior force agrees with directional finite differences and
   a second-order Taylor remainder, including the prior transformation. The
   likelihood force \(\nabla_z\Phi\) used by the integrator's kicks also agrees,
   on its own, with directional finite differences of \(\Phi\).
4. The split integrator is reversible and volume-preserving, and its energy
   error over a fixed trajectory duration scales as \(\varepsilon^2\). For the
   prior-only target (\(\Phi=0\)), energy error is zero to round-off and
   acceptance is unity to that accuracy. Rejected trajectories preserve their
   starting state.
5. HMC reproduces the mean and covariance of an analytically known Gaussian
   target, including a correlated case and prior-only sampling.
6. Simulation-based calibration (Talts et al. 2018): draw truths and data from
   the same declared discrete model used for inference, then test posterior
   ranks of scalar summaries including spatial mean and selected nodal values.
   Account for autocorrelation in HMC draws and finite-experiment uncertainty
   when assessing uniformity. This requires many independent posterior runs
   and is performed on the smallest validation mesh. Fine-mesh truth with
   coarse-mesh inference is a separate approximation check, not this test of
   sampling a correctly specified discrete model.
7. A small full-field heat problem runs end to end using fine-mesh synthetic
   observations and a coarser inference mesh. Tightening the PDE tolerance
   leaves the potential and force stable at the chosen sampling accuracy.
   Across meshes with the same sensors, measure acceptance, energy errors,
   autocorrelation, and effective independent draws per unit cost, with Monte
   Carlo uncertainty. Monotonically nondecreasing acceptance is not a pass
   condition.
8. Prior adequacy for an interface: with data generated on a finer mesh from
   `_sharp_interface_h` in `src/benchmark_utils.py`, report the fraction of
   nodes at which the 90% credible interval contains the true conductivity,
   separately for a stated physical neighborhood of the interface and for the
   remaining nodes. This is a modelling check, not a sampler test; it assesses
   the limitations of the Gaussian baseline under the accepted Q6 science case.
   Exact truth recovery or nominal coverage for a misspecified prior is not a
   pass condition for sampler correctness. A fraction of nodes covered in one
   field is a spatial diagnostic; repeated noise realizations at fixed physical
   evaluation locations are needed to assess repeated-experiment coverage.
9. The new optimization and sampling calculations evaluate the same posterior.
   Uniform physical initialization retains all field unknowns. Original nodal
   observations preserve positions, ordering, values, and uncertainties. Returned
   draws retain their physical spatial association and can be plotted directly.
   Later mutation of the supplied source, boundary data, or observation arrays
   leaves the declared inference problem unchanged. Default four-chain runs,
   explicit single-chain checks, per-chain counts, reproducible dispersed
   starts, and physical summaries from saved draws have the agreed meanings.
10. New inference and prediction calls restore the supplied conductivity and
    temperature on success and handled interruption/failure. Explicit application
    of a selected field recomputes its matching temperature; a mismatched
    destination model is rejected before modification. Its current conductivity
    and temperature are not treated as physical-model mismatches. Continuation
    from a saved trajectory boundary, including warmup, reproduces an uninterrupted
    seeded run in the same environment, including rejection repeats and
    diagnostic counts; incompatible settings are rejected. Uneven chain
    interruption resumes to the original per-chain total, and explicit extension
    reaches a new total without repeating completed warmup or altering old draws.
11. The existing tests and documentation checks continue to pass.

Interview Q3 accepted staged validation, and Q7 specified tests at `nmesh=16`,
`32`, and `64`. The current square meshes have 289, 1089, and 4225 nodal
conductivity unknowns respectively. Compare the same physical observations
across these inference meshes and record sampling diagnostics and cost.
Q11 requires numerical validation followed by smooth-truth and sharp-variation
studies under the Gaussian baseline at these resolutions. Use four independently
seeded, dispersed chains for each reported posterior. Each heat solve initially
uses one process; independent chains may run separately. Pilot timings inform
run lengths, and results report elapsed time, heat/adjoint solve counts, and
effective draws per unit cost, including unconverged cases. No runtime cap or
guarantee of convergence within an arbitrary budget has been specified. Prior
parameters remain explicit, and sensitivity runs must not tune them to force
recovery of a known synthetic truth.

## Qualification of posterior summaries

A short chain establishes execution, not convergence. Q15 requires all of the
following for the initial scientific demonstrations:

- Four-chain `R-hat < 1.01`, using the maximum of rank-normalized split and
  folded-split statistics.
- Combined bulk ESS and tail ESS at least 400 for each monitored quantity.
  These are four-chain diagnostics, not 400 effective draws required per chain
  or a sum of separately estimated single-chain ESS values.
- Zero recorded post-warmup divergences under the documented numerical rule.

Monitor every nodal conductivity and the reported nonconstant physical
quantities, including spatial mean, spatial contrasts, and sensor predictions.
Report Monte Carlo errors for the means and interval endpoints being used,
worst diagnostics and their spatial locations, and posterior predictive
residuals. Identify analytically fixed quantities, such as prescribed boundary
temperatures, separately. Undefined or nonfinite diagnostics for inferred
quantities cannot pass by omission. Missing diagnostics likewise cannot qualify
a result. Retain and clearly label runs that have not met these thresholds.

These are practical screening thresholds, not proof of global posterior
exploration or sufficient precision for every scientific question. Numerical
force/trajectory checks and prior adequacy remain independent requirements.
Mesh-refinement sensitivity is needed for a continuum interpretation; success
on one mesh alone does not establish it. In the sharp-truth study, favorable
chain diagnostics do not establish that a Gaussian prior describes the material
interface accurately.

## References

- [Lindgren, Rue & Lindström (2011), the SPDE construction of Gaussian fields](https://doi.org/10.1111/j.1467-9868.2011.00777.x).
- [Stan reference manual: HMC and adaptation](https://mc-stan.org/docs/reference-manual/mcmc.html).
- [Stan reference manual: change-of-variables determinants](https://mc-stan.org/docs/reference-manual/transforms.html).
- [Stan: interpretation of sampling diagnostics](https://mc-stan.org/learn-stan/diagnostics-warnings.html).
- [RStan: rank-normalized diagnostics and bulk/tail effective sample sizes](https://mc-stan.org/rstan/reference/Rhat.html).
- [Vehtari et al. (2021), rank-normalization, folding, and localization](https://arxiv.org/abs/1903.08008).
- [Beskos, Pinski, Sanz-Serna & Stuart (2011), Hybrid Monte Carlo on Hilbert spaces](https://doi.org/10.1016/j.spa.2011.06.003).
- [Beskos, Pillai, Roberts, Sanz-Serna & Stuart (2013), optimal tuning of the hybrid Monte Carlo algorithm](https://doi.org/10.3150/12-BEJ414).
- [Neal (2011), MCMC using Hamiltonian dynamics](https://arxiv.org/abs/1206.1901).
- [Knapik, van der Vaart & van Zanten (2011), Bayesian inverse problems with Gaussian priors](https://doi.org/10.1214/11-AOS920).
- [Talts, Betancourt, Simpson, Vehtari & Gelman (2018), simulation-based calibration](https://arxiv.org/abs/1804.06788).
- [Roininen, Girolami, Lasanen & Markkanen (2019), hyperpriors for Matérn fields](https://doi.org/10.3934/ipi.2019001).
- [Iglesias, Lu & Stuart (2016), a Bayesian level set method for geometric inverse problems](https://doi.org/10.4171/IFB/362).
- [Cotter, Roberts, Stuart & White (2013), MCMC methods for functions](https://doi.org/10.1214/13-STS421).

## Requirements review

Q1–Q24 are accepted and incorporated. There are no outstanding identified
requirements questions. Following the request on 2026-10-26, the
[numerical implementation plan](../plans/2026-10-26-full-field-blackjax-hmc.md)
records the concrete interfaces, tests and scientific qualification work for review.
Routine names, storage format, recorded example parameter values,
diagnostic-package selection, and pilot-based chain lengths are implementation
responsibilities within the agreed scope. Accepted library and notebook
requirements do not constitute numerical qualification.

The first release does not infer prior/noise hyperparameters, add correlated
noise, implement all future sharp-field priors, or distribute one heat solve
over MPI ranks. It preserves the physical model and observations needed for
those later studies. No HMC code or new sampling-validation result is claimed
by completion of this specification.
