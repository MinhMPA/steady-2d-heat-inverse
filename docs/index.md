# steady-2d-heat-inverse

Recover a spatially varying thermal conductivity `h(x,y)` from noisy steady-state
temperature measurements, using the heat equation as the physical model.

## Optimization and sampling

The conductivity posterior combines fixed physical thermometers, independent
Gaussian measurement errors and a proper Gaussian prior on log conductivity.
Two complementary estimates describe this same posterior:

- **Optimization** finds a mode in log conductivity and maps it to a positive
  conductivity field.
- **Sampling** explores conductivity fields drawn from the posterior, giving a
  mean reconstruction and uncertainty at each location.

Both methods use the **discrete adjoint method** for the same data-misfit gradient.
One forward heat solve and one adjoint solve give the gradient with respect to the
entire conductivity field, independently of its number of unknowns. Sampling uses
**Hamiltonian Monte Carlo (HMC)**: auxiliary momenta and adjoint gradients guide
trajectories through the posterior. The custom split-Hamiltonian calculation uses
BlackJAX components; each trajectory requires many gradient evaluations.

Start with the [common-posterior tutorial](hmc.md),
[saved fields and continuation](hmc-results.md), and
[current scientific evidence](validation/hmc-status.md).
The [legacy TAO example](#legacy-tao-example) below retains its original,
different regularized objective.

The optimization and sampling interfaces are present on the
`docs/hmc-inference-guide` implementation branch. A local documentation build
does not establish their availability on `master` or the live Read the Docs site.

```{figure} _static/figures/conductivity-inference.svg
:alt: Conductivity recovered by optimization and sampling, with truth and pointwise credible intervals

Conductivity recovered by optimization and sampling. Shading shows pointwise
68% and 95% credible intervals.
```

[Open full-size figure](_static/figures/conductivity-inference.svg).

Two central slices show the physical-conductivity posterior mean (blue), the
log-conductivity mode candidate (orange), and the generating truth. Darker
shading marks 68% and lighter shading 95%. This example uses four chains of
1,024 retained draws on mesh 32, with truth generated on mesh 16; see the
[pointwise interpretation and qualification](validation/hmc-status.md#chain-convergence).
Download the {download}`plotted values <_static/figures/conductivity-slices.csv>`,
{download}`calculation record <_static/figures/conductivity-slices.json>` or
{download}`redrawing script <../scripts/plot_inference_example.py>`.

The heat equation is solved on the unit square with a fixed bottom-wall
temperature and insulated remaining walls. [FEniCSx](https://fenicsproject.org/)
supplies the finite-element discretization; [PETSc](https://petsc.org/)
supplies the heat and adjoint linear solves and legacy
TAO optimization. Sampling uses one MPI rank in its qualified environment.

## Forward and inverse problems

```{figure} ../assets/forward-problem.svg
:alt: Known conductivity mapped to a predicted temperature field by the heat equation
:width: 100%

Predict the temperature from the conductivity.
```

```{figure} ../assets/inverse-problem.svg
:alt: Noisy temperature measurements mapped to a conductivity reconstruction by adjoint-based TAO optimization
:width: 100%

Recover the conductivity from temperature measurements: the legacy optimization route.
```

The [forward model](theory.md#forward-problem) supplies the temperature prediction
for both optimization and sampling. The inverse illustration shows the
[legacy TAO reconstruction](theory.md#legacy-reconstruction-objective); the
posterior example above also quantifies uncertainty through sampling.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
theory
usage
testing
hmc
hmc-results
validation/hmc-status
validation/hmc-numerics
validation/hmc-mesh-study
validation/hmc-mpi-shutdown
api/index
```

```{toctree}
:maxdepth: 1
:caption: Scientific decisions

adr/0001-share-the-conductivity-posterior
adr/0002-allow-priors-for-sharp-conductivity
adr/0003-preserve-the-supplied-physical-state
```

## Legacy TAO example

This original reconstruction minimizes a temperature-field least-squares misfit
with an H¹ smoothness penalty and a positivity bound. Its objective differs from
the fixed-thermometer posterior above. See [legacy usage](usage.md) and
[theory](theory.md#legacy-reconstruction-objective) for its retained conventions.

```python
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver

# Synthesize a noisy observation from a known conductivity.
truth = SteadyHeat2DForwardSolver(
    nmesh=64, h=lambda x: 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2), q=1.0
)
truth.solve()
T_obs = truth.add_noise(0.0, 1e-3, seed=0)

# Recover h(x,y) from a different initial guess. The guess must be spatially varying:
# TAO optimizes the DOF vector of `fwd.h.function`, and a scalar h becomes a fem.Constant
# with no DOFs to optimize.
fwd = SteadyHeat2DForwardSolver(
    nmesh=64, h=lambda x: 2.0 + 3.0 * x[0] ** 2 + x[0] / (4.0 + 3.0 * x[1] ** 2), q=1.0
)
fwd.solve()
# Pass DOF values, not the Function: `truth` and `fwd` are separate solver instances and
# so own separate function spaces, which the adjoint solver asserts against.
adj = SteadyHeat2DAdjointSolver(fwd, T_obs.x.array, sigma=1e-3, alpha=5e-3, DBC_value=0.0)
adj.solve()
h_sol = SteadyHeat2DTAOSolver(fwd, adj).solve()
```

:::{note}
The sources are installed as **flat top-level modules**, so the import is
`from forward_solver import ...` — never `from src.forward_solver import ...` and never a
`steady_2d_heat_inverse` package. See {doc}`installation`.
:::

## References

- Hans P. Langtangen and Kent-Andre Mardal,
  [*Introduction to Numerical Methods for Variational Problems*](https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf)
- Hans P. Langtangen and Anders Logg (adapted by Jørgen S. Dokken),
  [*The FEniCS tutorial*](https://jsdokken.com/dolfinx-tutorial/)
- Hans P. Langtangen,
  [*Approximation of Functions*](https://hplgit.github.io/num-methods-for-PDEs/doc/pub/approx/pdf/approx-4print.pdf)
- Andrew M. Bradley,
  [*PDE-constrained optimization and the adjoint method*](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)

## Indices

- {ref}`genindex`
- {ref}`modindex`
