# Conductivity posterior from fixed thermometers

The heat equation is $-\nabla\cdot(h\nabla T)=q$ on the unit square, with a
300 K bottom wall and insulated remaining walls. We infer every conductivity node
on the chosen P1/Q1 mesh. Full-field inference describes this discrete posterior;
continuum accuracy requires a separate mesh study.

The executable example is
[HMCSampling.ipynb](https://github.com/MinhMPA/steady-2d-heat-inverse/blob/master/notebooks/HMCSampling.ipynb).
Its default `nmesh=16`, four chains, 100 warmup iterations and 200 retained draws per
chain are **illustrative and unqualified**. The data use `fine_nmesh=64`, 256 fixed
interior thermometers and one seeded noise realization. The planned scientific
report is `docs/validation/hmc-mesh-study.md`; it has no results yet. This path will
become a report link when the scientific study is available.

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

## One declared physical experiment

```python
from conductivity_prior import GaussianLogConductivity
from heat_inference import HeatInverseProblem
from heat_model import HeatNumerics, NodalField
from hmc_blackjax import HMCControls
from temperature_observations import TemperatureObservations

observations = TemperatureObservations(sensor_points, measured_kelvin, sigma=1e-3)
problem = HeatInverseProblem(
    forward=forward, observations=observations,
    prior=GaussianLogConductivity(mu=0, kappa=2, tau=1, h_ref=1),
    numerics=HeatNumerics(),
)
estimate = problem.optimize(initial_h=1.0)
print(estimate.success, estimate.message)
samples = problem.sample(
    initial=estimate, seed=7, output="runs/example"
)
```

`HeatInverseProblem` freezes the mesh, heat source, boundary temperatures,
measurements, prior and numerical tolerances. The caller's current conductivity
and temperature do not define the experiment and remain unchanged by optimization,
sampling, private temperature predictions and summaries. Every heat solve uses one
MPI rank, CPU float64 and fixed tolerances. A failed heat solve stops the run with
its diagnostic. Nonfinite proposals repeat the last valid state and count as
divergences. The legacy adaptive heat-solve tolerances are not used here.

A scalar, callable or `NodalField(values, coordinates)` can supply an initial
conductivity. Nodal coordinates must match the inference mesh; there is no automatic
cross-mesh transfer. `problem.initial_state(value)` constructs a complete inferred
state. Pass `initial=estimate`, rather than extracting conductivity alone: this
preserves every inferred coordinate. Optimization reports the mode of the density
in **log conductivity** as a physical field. Check its `success` and `message`;
a failed search explicitly returns its last finite state, not a claimed mode.

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
from hmc_blackjax import HMCControls

starts = [problem.initial_state(value) for value in (0.9, 1.0, 1.1, 1.2)]
samples = problem.sample(
    initial_states=starts, chains=4, seed=7, warmup=1000, draws=2000,
    controls=HMCControls(
        initial_step_size=5e-4, step_size_bounds=(1e-6, 1e-3),
        target_acceptance=0.8,
        trajectory_interval=(0.7853981633974483, 2.356194490192345),
        initial_spread=0.3,
    ),
)
```

Supply either `initial` or `initial_states`. Explicit states are used exactly;
`initial_spread` applies to a shared center with multiple chains. A single-chain
execution check is explicit: `chains=1, warmup=8, draws=12`. It cannot establish
four-chain convergence.

The dynamics combine likelihood-force kicks with exact Gaussian-prior rotations,
identity momentum covariance and randomized trajectory durations. Only step size
adapts during warmup. Retained draws use frozen controls. Zero warmup requires
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

## Saved fields, errors and physical interpretation

```python
report = samples.diagnostics()  # optional [diagnostics] extra; saves an advisory report
summary = samples.summary()    # uses a matching report; does not run diagnostics
h_mean, h_sd = summary.mean, summary.sd
pointwise_90 = summary.quantiles[[0, 2]]  # defaults: 0.05, 0.5, 0.95
print(summary.counts, summary.qualified)
h_one = samples.field(chain=0, draw=0)
T_one = samples.temperature(chain=0, draw=0)
predicted_kelvin = samples.sensor_predictions(chain=0)
residual = (predicted_kelvin.mean(axis=0) - observations.values) / observations.sigma
subset = samples.load_draws(chains=[0], start=0, stop=10)
```

Fields have immutable physical coordinate labels and cell connectivity. Full
`load_draws()` allocates `(chain, draw, node)` arrays explicitly; summaries read
bounded node chunks from disk. Every retained iteration contributes, including
rejection repeats. `summary.counts` gives committed counts per chain. Summaries
pool available states, so unequal partial chains contribute unequal weight and
cannot qualify the requested calculation.

The physical transformation precedes statistics: $E[h]$ generally differs from
$h_{ref}\exp(E[m])$. Pointwise credible intervals are not simultaneous field bands.
Likewise $T(E[h])$ generally differs from $E[T(h)]$. Average temperatures from saved
conductivity draws to estimate posterior mean temperature. Saved thermometer
predictions already supply this averaging at sensor locations. These are latent
temperature predictions; a future measurement also has measurement noise.

Convergence requires four chains, rank-normalized split/folded $\widehat R<1.01$,
combined bulk and tail ESS at least 400 for every monitored nonconstant quantity,
and zero retained divergences. Inspect `report.per_quantity` for mean/quantile
Monte Carlo errors and `report.worst_locations` for spatial weaknesses. Missing or
nonfinite diagnostics cannot pass; a successful execution is not convergence
evidence. If the optional diagnostic package is absent, the report remains
explicitly unqualified with reasons.

Chain convergence, numerical heat-solve accuracy, mesh approximation and prior
adequacy are separate conclusions. Analytically prescribed boundary thermometers
have zero stochastic uncertainty. Their interpolated values may still have raw
residuals: `report.fixed_sensor_consistency` preserves their ranges and deviations,
and labels every nonzero discrepancy unverified. Convergence qualification does
not establish numerical consistency for those sensors.

## Physical figures and explicit application

The notebook uses optional Matplotlib with recorded coordinates and connectivity,
then exports small `illustrative-fields.npz`, `conductivity.png` and
`temperature-residuals.png` files inside the run directory. A separate plotting
environment can load the NumPy archive. Keep the FEM environment intact: the
PyVista/VTK extra has a known native-library conflict in the current environment.
An optional, separately qualified existing plot route is

```python
from plotting_utils import plot_scalar_mesh
f = samples.as_function(summary="mean")
plot_scalar_mesh(f.function_space, f.x.array, "h")
```

The independent triangle/quadrilateral node-association test checks this mapping;
it does not qualify an installed PyVista/VTK environment. Explicitly applying a
selected field checks the destination mesh, source and boundaries:

```python
T_at_mean_h = samples.apply_to(forward, summary="mean")
# Or samples.apply_to(forward, chain=0, draw=0)
```

This updates the caller's conductivity and corresponding temperature. The result
is $T(E[h])$ for `summary="mean"`. Standard deviation is not an applicable physical
conductivity. Current caller conductivity/temperature values are excluded from the
experiment-compatibility comparison.

## Interruption, continuation and storage

The output must be a new directory; the default is a unique directory under
`runs/`. It contains `problem.json`, `problem.npz`, `chains.sqlite` and advisory
`status.json`. Diagnostics add `diagnostics.json`. SQLite records every warmup and
retained iteration, random state, adaptation and continuation state transactionally.
Long histories belong in run directories rather than notebook outputs. Close
results and inference problems when finished, or use their context managers.

```python
with HeatInverseProblem.load("runs/example") as restored:
    with restored.resume("runs/example") as result:
        print(result.summary().counts)  # finish the original per-chain target
    with restored.resume("runs/example", total_draws=4000) as extended:
        report = extended.diagnostics()  # old qualification no longer applies
```

After interruption, resume recovers the last committed trajectory. Explicit
extension raises the target rather than changing the scientific problem. Changed
physics, numerical settings or recorded environment cannot append to the run.
Copy a closed run directory or use SQLite's backup operation; copying an active
database alone is not a backup.

The original integrated-misfit, roughness-regularized reconstruction remains
available with its original mathematics. Its regularization parameter and objective
do not become this fixed-sensor posterior's noise law or prior.

## Smooth and sharp physical comparisons

The [mesh and interface study](validation/hmc-mesh-study.md) freezes one refined
observation realization for each truth, then uses identical physical measurements
at inference meshes 16, 32 and 64. `make_truth_data`, `run_mesh_study`,
`interface_coverage`, and `write_study_report` are available from `hmc_validation`.
Each actual observation set and fixed prior requires its matching numerical pilot.
Eight predeclared n16 prior variations are separate posteriors. The report retains
missing and failed outcomes, distinguishes Monte Carlo errors from mesh/prior
changes, and includes observation noise in predictive residuals.
