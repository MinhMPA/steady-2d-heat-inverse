# Conductivity posterior from fixed thermometers

The heat equation is $-\nabla\cdot(h\nabla T)=q$ on the unit square, with a
300 K bottom wall and insulated remaining walls. We infer every conductivity node
on the chosen P1/Q1 mesh. Full-field inference describes this discrete posterior;
continuum accuracy requires a separate mesh study.

The executable example is
{download}`HMCSampling.ipynb <../notebooks/HMCSampling.ipynb>`, included from
the same source revision as this guide. Sampling is available on the
`docs/hmc-inference-guide` branch; availability on `master` or the hosted site
is not established by this local build.
Its default `nmesh=16`, four chains, 100 warmup iterations and 200 retained draws per
chain are **illustrative and unqualified**. The data use `fine_nmesh=64`, 256 fixed
interior thermometers and one seeded noise realization. The planned scientific
study is recorded in the [mesh and prior validation report](validation/hmc-mesh-study.md).
Its reference observations and exploratory pilots are available; production
calibration and prior-sensitivity calculations remain incomplete.

## Measurements and prior

Measurements are temperatures in kelvin at physical $(x,y)$ coordinates. Each has
an independent Gaussian error with a known positive standard deviation in kelvin:

$$\Phi(h)=\frac12\sum_s\left(\frac{[BT(h)]_s-d_s}{\sigma_s}\right)^2.$$

There is no mass matrix in this likelihood. A scalar `sigma` applies to every
thermometer; a vector supplies one known error per thermometer. Noise amplitude is
not inferred. The sensor positions and observed temperatures remain fixed across
inference meshes; only their finite-element prediction operator changes. Generate
synthetic data on a finer mesh to avoid using the same discretization for data and
inference. `TemperatureObservations.from_function(T_obs, sigma=...)` preserves the
original measurement nodes and values when importing nodal observations.

The proper Gaussian log-conductivity prior is

$$m=\log(h/h_{ref})=\mu+\tau^{-1}A^{-1}M_L^{1/2}z,\quad
A=\kappa^2M+K,\quad z\sim N(0,I),\quad h=h_{ref}e^m.$$

The likelihood and $\tfrac12z^Tz$ define the potential. There is no additional
conductivity Jacobian or legacy conductivity bound. The prior has natural Neumann
boundaries and positive `kappa`, `tau`, `h_ref`. `mu` is dimensionless;
`kappa` is an inverse physical length; `h_ref` uses conductivity units. `tau`
controls amplitude through this elliptic construction. All prior parameters stay
fixed. The example uses `mu=0, kappa=2, tau=1, h_ref=1`; these are illustrative
choices, rather than a validated prior for sharp material interfaces.

## A complete, deliberately short experiment

This example heats a unit square with source `q=1`, a 300 K bottom wall and
insulated remaining walls. Nine fixed thermometers observe a smooth conductivity
on a generating mesh with four subdivisions per side. Inference uses a separate,
coarser triangle mesh with two subdivisions per side (nine conductivity nodes).
One seeded Gaussian noise realization has known `sigma=0.05` K. These choices
make a small execution check; they do not qualify uncertainty or continuum accuracy.

Run from an installed checkout in the [qualified native environment](installation.md#separate-cpu-posterior-environment),
using one MPI rank. The output is a new disposable directory; keep the printed
path for the [saved-result examples](hmc-results.md). All variables are defined here.

```python
from pathlib import Path
import tempfile
import numpy as np

from conductivity_prior import GaussianLogConductivity
from forward_solver import SteadyHeat2DForwardSolver
from heat_inference import HeatInverseProblem
from heat_model import HeatNumerics
from hmc_blackjax import HMCControls
from temperature_observations import SensorOperator, TemperatureObservations

truth = SteadyHeat2DForwardSolver(
    nmesh=4, mesh_type="triangle",
    h=lambda x: 1 + 0.2*x[0] + 0.1*x[1], q=1.0, DBC_value=300.0,
)
truth.solve()
axis = (0.25, 0.5, 0.75)
sensor_points = np.array([(x, y) for x in axis for y in axis])
sigma = 0.05  # kelvin, independently for each thermometer
noiseless_kelvin = SensorOperator(truth.V, sensor_points).predict(truth.T.x.array)
measured_kelvin = noiseless_kelvin + np.random.default_rng(19).normal(
    scale=sigma, size=len(sensor_points)
)
observations = TemperatureObservations(sensor_points, measured_kelvin, sigma=sigma)
prior = GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1)
forward = SteadyHeat2DForwardSolver(
    nmesh=2, mesh_type="triangle", h=1.0, q=1.0, DBC_value=300.0,
)
forward.solve()
caller_h = np.array(forward.h.function.value, copy=True)  # this h is a Constant
caller_T = forward.T.x.array.copy()
output = Path(tempfile.mkdtemp(prefix="heat-demo-")) / "posterior"
probabilities = (0.025, 0.16, 0.5, 0.84, 0.975)

with HeatInverseProblem(
    forward=forward, observations=observations, prior=prior, numerics=HeatNumerics(),
) as problem:
    estimate = problem.optimize(
        initial_h=1.0, maxiter=500, gtol=1e-6, ftol=1e-12,
    )
    print("Optimization:", estimate.success, estimate.message)
    with problem.sample(
        initial=estimate, seed=7, chains=1, warmup=8, draws=12, output=output,
        controls=HMCControls(
            initial_step_size=0.1, step_size_bounds=(1e-6, np.pi/8),
            initial_spread=0.3,
        ),
    ) as samples:
        report = samples.diagnostics(quantiles=probabilities)
        summary = samples.summary(quantiles=probabilities)
        print("Sampling:", report.status, report.reasons)
        print("Summary:", summary.counts, summary.qualified)
        h_mean, h_sd = summary.mean, summary.sd
        pointwise_68 = summary.quantiles[[1, 3]]
        pointwise_95 = summary.quantiles[[0, 4]]
        xy, cells = summary.coordinates.copy(), summary.cells.copy()
        assert h_mean.shape == (len(xy),)
        assert np.array_equal(summary.probabilities, probabilities)
        assert np.array_equal(report.probabilities, probabilities)
        assert not report.qualified and not summary.qualified

assert np.array_equal(caller_h, forward.h.function.value)
assert np.array_equal(caller_T, forward.T.x.array)
print("Saved result:", output)
```

The example's large step size `0.1` is confined to this coarse, weakly constrained
experiment, following the small-mesh workflow check. It is **not a recommendation
for larger meshes or more precise thermometers**. One chain and twelve retained
draws remain unqualified even if trajectories finish without divergences. The
production defaults and their scientific limits are given separately below.

`HeatInverseProblem` freezes the mesh, heat source, boundary temperatures,
measurements, prior and numerical tolerances. Optimization and sampling use the
same posterior and preserve the caller's conductivity and temperature. Closing the
problem releases its private heat/adjoint workspaces; the caller still owns
`forward` and `truth`, which may be used again. Saved-result handles are closed
by their context managers. The forward solver has no public `close()` method;
its own finite-element objects retain their usual lifetime in the calling program.

A scalar, callable or `NodalField(values, coordinates)` (from `heat_model`) can
supply `initial_h`. Nodal coordinates must match the inference mesh; there is no
automatic cross-mesh transfer. `problem.initial_state(value)` constructs a complete
inferred state. Pass `initial=estimate` to preserve every inferred coordinate.
Optimization returns a `PosteriorEstimate`, whose `success` and `message` must be
read before calling it a mode: a failed search returns its last finite candidate.
A finite candidate may still initialize sampling; it is not a convergence claim.

Every heat solve uses CPU float64, one MPI rank and fixed tolerances. A failed
finite heat or adjoint solve stops the calculation with its diagnostic. Nonfinite
proposals repeat the last valid state and count as divergences. The legacy adaptive
heat-solve tolerances are not used here. For the measured macOS ARM64 MPICH
environment, declare TCP transport and the clean compiler settings **before**
starting Python, following [installation](installation.md#macos-mpich-transport-and-shutdown).
The [dated evidence](validation/hmc-status.md#numerical-reproducibility) records the
still-open limitation in identifying effective form-compiler settings.

## Chains and trajectories

`sample` defaults to four chains, 1000 warmup iterations and 2000 retained draws per
chain. One master seed determines independent streams. A common initial state is
dispersed in whitened coordinates with default spread `1.0`. With `controls=None`,
the initial integration spacing is `6.5e-4`, bounded by `(1e-6, 1.3e-3)` during
adaptation. These reference-informed settings come from the measured local
curvature and the three-mesh numerical pilot at source `40f1324`; spread `1.0`
is the largest passing choice among `0.1`, `0.3` and `1.0` in its mesh-16 pilot.
The evidence is conditional on the documented prior, data and mesh. It does not
guarantee stability for every physical problem or establish production
convergence. The final numerical revision requires its new matching pilot before
calibration. Explicit per-chain starts and custom controls use

```python
# Optional longer calculation: the short example above has already closed its problem.
with HeatInverseProblem(
    forward=forward, observations=observations, prior=prior, numerics=HeatNumerics(),
) as problem:
    starts = [problem.initial_state(value) for value in (0.9, 1.0, 1.1, 1.2)]
    with problem.sample(
        initial_states=starts, chains=4, seed=7, warmup=1000, draws=2000,
        controls=HMCControls(
            initial_step_size=5e-4, step_size_bounds=(1e-6, 1e-3),
            target_acceptance=0.8,
            trajectory_interval=(0.7853981633974483, 2.356194490192345),
            initial_spread=0.3,
        ),
    ) as longer:
        print(longer.output)  # inspect current diagnostics before scientific use
```

Supply either `initial` or `initial_states`. Explicit states are used exactly;
`initial_spread` applies to a shared center with multiple chains. A single-chain
execution check is explicit: `chains=1, warmup=8, draws=12`. It cannot establish
four-chain convergence.

The dynamics combine likelihood-force kicks with exact Gaussian-prior rotations,
identity momentum covariance and randomized trajectory durations. This is a
custom split-Hamiltonian calculation using BlackJAX component machinery, rather
than an unmodified default NUTS or stock HMC call. The existing discrete-adjoint
connection evaluates the PDE eagerly on CPU in float64. A likelihood-gradient
evaluation needs one heat solve and one adjoint solve, independently of the
number of conductivity unknowns; a trajectory evaluates many such forces. Only
step size adapts during warmup. Retained draws use frozen controls. Zero warmup requires
`adapt_step_size=False` and an explicitly supplied step size, even when that
value equals the library default. The randomized duration law remains
`(pi/4, 3*pi/4)`. The notebook uses explicit
initial spacing `5e-4` and upper bound `1e-3`, preserving the default randomized
duration interval. At `nmesh=4` the measured largest local posterior curvature was
$1.77787\times10^6$, with local harmonic spacing scale
$2/\sqrt{\lambda_{max}}=0.00149996$. The chosen ceiling is below that local scale;
global stability still needs actual trajectory evidence. The library's initial
former provisional spacing `0.1` caused two rejected divergent warmup trajectories and a subsequent
fatal adjoint failure at an extremely small positive finite conductivity. The
stopped calculation is retained as evidence. The new reference-informed defaults
replace that provisional spacing and ceiling. A failed finite heat solve still
stops the run. The notebook retains its independently chosen explicit controls.

## Physical uncertainty and qualification

The probabilities in the complete example request equal-tailed **pointwise 68%
and 95% credible intervals of physical conductivity**. Use the same probabilities
for `samples.diagnostics(quantiles=probabilities)` and
`samples.summary(quantiles=probabilities)`, so the quantile Monte Carlo errors and
summary qualification refer to the intervals being interpreted. Summary alone
does not run diagnostics. A missing, stale or incomplete matching diagnostic
report leaves `summary.qualified=False`.

Transform each draw to conductivity before averaging: $E[h]$ generally differs
from $h_{ref}\exp(E[m])$. The optimized result is the mode in **log conductivity**,
reported as a physical field; the sampling estimate is the mean of **physical
conductivity**. These need not coincide. Pointwise intervals describe each
location separately, not a simultaneous 95% envelope for the whole field.

Likewise $T(E[h])$ generally differs from $E[T(h)]$. Average temperatures evaluated
at individual draws to estimate posterior mean temperature; saved thermometer
predictions already allow that averaging at sensor positions. These are latent
temperature predictions. A future thermometer reading also includes measurement
noise.

Qualification requires four chains, rank-normalized split/folded
$\widehat R<1.01$, combined bulk and tail ESS at least 400 for every monitored
nonconstant quantity, and zero retained divergences. Inspect `report.per_quantity`
for mean/quantile Monte Carlo errors, `report.worst_locations` for spatial weaknesses,
and `report.reasons` for missing evidence. Missing or nonfinite diagnostics cannot
pass; absent optional diagnostic packages produce an explicitly unavailable,
unqualified report. The short example cannot meet the four-chain requirement.

Analytically prescribed boundary thermometers have zero stochastic uncertainty.
`report.fixed_sensor_consistency` separately retains raw interpolation residuals
and labels every nonzero discrepancy unverified. Chain convergence does not
establish that numerical consistency. Monte Carlo error, repeated-experiment
calibration, finite-element discretization error and prior adequacy are distinct
questions. A Gaussian prior can smooth sharp material interfaces even with
well-converged chains. The [16 November evidence summary](validation/hmc-status.md)
separates the qualified saved mesh-32 diagnostic snapshot from the incomplete
calibration and mesh/prior studies.

The [saved-result and continuation guide](hmc-results.md) covers physical field
arrays, full temperature predictions, figures, explicit application to a caller's
heat model, interrupted histories and deterministic closure. For exported plots,
label the curves **Sampling** and **Optimization** and distinguish the two
pointwise interval levels using one sampling color with two band opacities.

The [theory](theory.md#fixed-sensor-conductivity-posterior) derives the whitening
map, sensor adjoint and split Hamiltonian. The [public reference](api/heat_inference.md)
records physical array labels, method signatures, defaults and failure meanings.

The original integrated-misfit, roughness-regularized TAO reconstruction remains
available with its original mathematics. It uses a different objective: its
regularization parameter does not become this fixed-sensor posterior's noise law
or prior.

## Smooth and sharp physical comparisons

The [mesh and interface study](validation/hmc-mesh-study.md) freezes one refined
observation realization for each truth, then uses identical physical measurements
at inference meshes 16, 32 and 64. `make_truth_data`, `run_mesh_study`,
`interface_coverage`, and `write_study_report` are available from `hmc_validation`.
Each actual observation set and fixed prior requires its matching numerical pilot.
Eight predeclared n16 prior variations are separate posteriors. The report retains
missing and failed outcomes, distinguishes Monte Carlo errors from mesh/prior
changes, and includes observation noise in predictive residuals.
