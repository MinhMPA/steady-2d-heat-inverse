# steady-2d-heat-inverse

Infer the spatially-varying thermal conductivity $h(x,y)$ of a 2D unit square from a
noisy steady-state temperature field, using a **discrete adjoint** gradient driven by
**PETSc TAO** bound-constrained optimization.

The package is built on [FEniCSx](https://fenicsproject.org/) (DOLFINx / UFL / basix) for
the finite-element discretization and on [PETSc](https://petsc.org/) for both the linear
solves (KSP) and the optimization (TAO). Every solver is MPI-aware.

## What it does

1. **Forward problem** — solve $-\nabla\cdot(h\nabla T) = q$ on $[0,1]^2$ with P1 Lagrange
   elements, a Dirichlet condition on the bottom wall and insulated (zero-flux Neumann)
   conditions elsewhere. Optionally inject Gaussian sensor noise to synthesize an
   observation $T_{\rm obs}$.
2. **Inverse problem** — recover $h(x,y)$ by minimizing a regularized least-squares
   misfit subject to the PDE and a positivity bound, using an adjoint gradient that costs
   ~2 linear solves regardless of the number of degrees of freedom.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
theory
usage
testing
api/index
```

## Quick example

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
