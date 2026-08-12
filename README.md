# steady-2d-heat-inverse

PDE-constrained inverse problem: infer the spatially-varying thermal conductivity `h(x,y)`
of a 2D unit square from a noisy steady-state temperature field, using a **discrete
adjoint** gradient driven by **PETSc TAO** bound-constrained optimization.

1. **Forward** — solve the 2D steady-state heat equation `-∇·(h∇T) = q` on `[0,1]²` with
   P1 Lagrange elements (DOLFINx): Dirichlet `T(y=0) = 300 K`, insulated (zero-flux
   Neumann) on the other three walls.
2. **Inverse** — recover `h(x,y)` by minimizing

   ```
   J[T(h),h] = ½ [ ∫(T − T_obs)²/σ² + α ∫|∇h|² ]
   ```

   subject to the PDE and the positivity bound `h ≥ h_min`. The `α` term is Tikhonov (H¹)
   regularization enforcing smoothness of `h`.

The adjoint state `λ` solves `−∇·(h∇λ) = (T − T_obs)/σ²` with `λ(y=0) = 0`, giving
`dJ/dh = −∇T·∇λ + α∇h·∇v`. The operator is self-adjoint, so the adjoint left-hand side is
identical to the forward one — **one gradient costs ~2 linear solves regardless of the
number of DOFs**. By default the optimization runs in `m = log h`, which enforces
positivity and applies the chain rule `dJ/dm = h · dJ/dh`.

## Installation

The scientific stack (DOLFINx, PETSc, MPI) is conda-only. With `conda`/`mamba` and `pip`:

```bash
conda env create -n steady-2d-heat-inverse -f environment.yml
conda activate steady-2d-heat-inverse
pip install -e ".[dev]"      # omit [dev] if you do not need pytest
```

The sources are installed as **flat top-level modules**, so imports are
`from forward_solver import ...` — never `from src.forward_solver import ...`.

## Usage

```python
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver

# Synthesize a noisy observation from a known conductivity.
truth = SteadyHeat2DForwardSolver(
    nmesh=128, h=lambda x: 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2), q=1.0
)
truth.solve()
T_obs = truth.add_noise(0.0, 1e-3, seed=0)

# Recover h(x,y) from a different initial guess. The guess must be spatially varying:
# TAO optimizes the DOF vector of `fwd.h.function`, and a scalar h becomes a fem.Constant
# with no DOFs to optimize.
fwd = SteadyHeat2DForwardSolver(
    nmesh=128, h=lambda x: 2.0 + 3.0 * x[0] ** 2 + x[0] / (4.0 + 3.0 * x[1] ** 2), q=1.0
)
fwd.solve()
# Pass DOF values, not the Function: `truth` and `fwd` are separate solver instances and
# so own separate function spaces, which the adjoint solver asserts against.
adj = SteadyHeat2DAdjointSolver(fwd, T_obs.x.array, sigma=1e-3, alpha=5e-3, DBC_value=0.0)
adj.solve()
h_sol = SteadyHeat2DTAOSolver(fwd, adj, verbose=1).solve()
```

All solvers are MPI-aware:

```bash
mpirun -n 4 python script.py
```

### Notebooks

- [notebooks/ForwardSolve.ipynb](notebooks/ForwardSolve.ipynb) — forward solve, noise
  injection, XDMF export of the synthetic "measured" data.
- [notebooks/InverseSolve.ipynb](notebooks/InverseSolve.ipynb) — full reconstruction from
  `test_data/blackbox_output.h5`.
- [notebooks/EvaluateSolution.ipynb](notebooks/EvaluateSolution.ipynb) — σ × α
  regularization sweep, Fourier transfer function `T(k)`, reconstruction-error histograms.

## Layout

```
src/
  forward_solver.py        SteadyHeat2DForwardSolver — mesh, weak form, solve, noise, XDMF, plots
  adjoint_solver.py        SteadyHeat2DAdjointSolver — adjoint weak form, solve, assemble_gradient
  tao_solver.py            SteadyHeat2DTAOSolver — TAO wrapper, bounds, log(h) reparam
  domain_coefficient.py    BaseDomainCoefficient + ThermalConductivity / HeatSource
  plotting_utils.py        plot_scalar_mesh() — PyVista rendering (rank-0 only)
tests/                     three independent gradient verifications
notebooks/                 ForwardSolve, InverseSolve, EvaluateSolution
test_data/                 blackbox_output.{xdmf,h5} — synthetic "measured" data
docs/                      Sphinx documentation source
```

## Tests

Gradient correctness is verified three independent ways — against central finite
differences, against a tangent-linear (JVP) solve, and via a Taylor-remainder convergence
rate of ≈ 2:

```bash
pytest -m gradcheck
```

These run real FEM solves and take minutes.

## Documentation

Full documentation (theory, usage, API reference) is built with Sphinx and configured for
[Read the Docs](https://about.readthedocs.com/):

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

The docs build mocks the FEniCSx/PETSc stack (`autodoc_mock_imports`), so it needs no
conda environment.

## References

- [1] Hans P. Langtangen and Kent-Andre Mardal, ["Introduction to Numerical Methods for Variational Problems"](https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf)
- [2] Hans P. Langtangen and Anders Logg (adapted by Jørgen S. Dokken) ["The FEniCS tutorial"](https://jsdokken.com/dolfinx-tutorial/)
- [3] Hans P. Langtangen, ["Approximation of Functions"](https://hplgit.github.io/num-methods-for-PDEs/doc/pub/approx/pdf/approx-4print.pdf)
- [4] Andrew M. Bradley, ["PDE-constrained optimization and the adjoint method"](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)
