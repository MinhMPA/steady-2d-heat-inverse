# steady-2d-heat-inverse

PDE-constrained inverse problem: infer the spatially-varying thermal conductivity `h(x,y)`
of a 2D unit square from a noisy steady-state temperature field, using a **discrete adjoint**
gradient driven by **PETSc TAO** bound-constrained optimization.

## Science

- **Forward model**: steady-state Poisson heat equation `-∇·(h∇T) = q` on `[0,1]²`.
  Dirichlet `T(y=0) = 300 K`; insulated (zero-flux Neumann) on the other three walls.
  P1 Lagrange elements on a quadrilateral or triangular mesh (DOLFINx).
- **Inverse problem**: minimize
  `J[T(h),h] = ½ [ ∫(T(h) − T_obs)²/σ² + α ∫|∇h|² ]`
  over `h`, subject to the PDE and the positivity bound `h ≥ h_min`.
  The `α` term is Tikhonov (H¹) regularization enforcing smoothness of `h`.
- **Gradient**: adjoint state `λ` solves `∇·(h∇λ) = −(T − T_obs)/σ²` with `λ(y=0)=0`; then
  `dJ/dh = −∇T·∇λ + α∇h·∇v`. The operator is self-adjoint, so the adjoint LHS is identical
  to the forward LHS — one gradient costs ~2 linear solves regardless of the number of DOFs.
- **Parametrization**: optimize in `m = log h` by default (`use_logh=True`), which enforces
  positivity and applies the chain rule `dJ/dm = h · dJ/dh` (`tao_solver.py:242`).

## Tech Stack

| Layer | Technology |
|---|---|
| FEM | DOLFINx + UFL + basix (FEniCSx) |
| Linear algebra / optimizer | PETSc (KSP: `cg` + `hypre`) and TAO (`blmvm`, `bncg`) via petsc4py |
| Parallelism | MPI via mpi4py (single-program, ghost-aware) |
| Interpolation | SciPy `RBFInterpolator` / `CloughTocher2DInterpolator` |
| Visualization / IO | PyVista, XDMF+HDF5, h5py |
| Env / build | conda-forge (`environment.yml`) + setuptools editable install |
| Tests | pytest with a `gradcheck` marker |

## Layout

```
src/                       flat top-level modules (see import gotcha below)
  forward_solver.py        SteadyHeat2DForwardSolver — mesh, weak form, solve, noise, XDMF export, plots
  adjoint_solver.py        SteadyHeat2DAdjointSolver — adjoint weak form, solve, assemble_gradient
  tao_solver.py            SteadyHeat2DTAOSolver — TAO wrapper, bounds, log(h) reparam, _objgrad callback
  domain_coefficient.py    BaseDomainCoefficient (ABC) + ThermalConductivity / HeatSource
  plotting_utils.py        plot_scalar_mesh() — PyVista rendering (rank-0 only)
tests/                     three independent gradient verifications + private helpers
notebooks/                 ForwardSolve, InverseSolve, EvaluateSolution
test_data/                 blackbox_output.{xdmf,h5} — synthetic "measured" data
docs/                      Sphinx source (MyST + autodoc); `.readthedocs.yaml` at repo root
```

**Import gotcha (most common mistake):** `pyproject.toml` installs the sources as *flat
top-level modules* via `py-modules`, so it is `from forward_solver import ...` — never
`from src.forward_solver` and never a `steady_2d_heat_inverse` package.
`src/__init__.py` is vestigial: its relative imports are never exercised and its `__all__`
misspells `SteadyHeat2DForwardSolver` as `SteadyHeatForwardSolver2D`.

## Data Flow

`h₀` → `ThermalConductivity` coerces input (scalar / callable / `fem.Expression` /
tabulated `(N,3)` array or DataFrame) into a `fem.Constant` or interpolated `fem.Function`
→ forward `LinearProblem` solves for `T` → adjoint solves for `λ` → `assemble_gradient()`
returns `dJ/dh` as a `PETSc.Vec` → TAO's `_objgrad` callback returns `(J, G)` → TAO proposes
a new `h` → `_update_h()` writes it back into the *shared* `fwd.h.function`, which both the
forward and adjoint UFL forms reference by handle. Loop until converged.

## Commands

```bash
conda env create -n steady-2d-heat-inverse -f environment.yml
conda activate steady-2d-heat-inverse
pip install -e ".[dev]"

pytest -m gradcheck          # adjoint gradient verification (real FEM solves)
mpirun -n 4 python script.py # MPI-parallel run

pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html   # docs build; no conda stack needed
```

## Conventions

- **Naming**: `snake_case` modules/functions, `SteadyHeat2D<Role>Solver` classes,
  leading `_` for test-private modules (`tests/_helpers.py`, `tests/_tangent_solver.py`)
  and internal methods.
- **Formatting**: `black` (the codebase was explicitly reformatted with it).
- **Imports**: grouped under comment banners (`# numerical imports`, `# dolfinx imports`,
  `# local imports`) — match this when editing.
- **Docstrings**: raw strings (`r"""`) with LaTeX-style math; every public method documents
  Parameters and Returns.
- **MPI hygiene**: after mutating `.x.array`, call `scatter_forward()`; after assembling a
  vector, `ghostUpdate()`. Print and plot only on `MPI.COMM_WORLD.rank == 0`.
- **Commits**: short imperative subject lines, no prefixes/scopes ("Clean up adjoint_solver").
- **Tests**: `tests/test_grad_*.py`, marked `pytest.mark.gradcheck` via a module-level
  `pytestmark`; markers are registered in `pyproject.toml`.

## Testing Strategy

Gradient correctness is verified three independent ways — keep all three passing when
touching `adjoint_solver.py` or the objective:

1. `test_grad_finitediff.py` — adjoint directional derivative vs. central finite difference.
2. `test_grad_forwarddiff.py` — vs. a tangent-linear (JVP) solve from `_tangent_solver.py`.
3. `test_grad_taylorexp.py` — Taylor-remainder convergence rate must be ≈ 2.

CI: `.github/workflows/install-and-import.yml` (env + editable install + import smoke test,
every push) and `run-gradchecks.yml` (micromamba + `pytest -m gradcheck`, on `src/**`,
`tests/**`, `pyproject.toml`, `environment.yml` changes).

## Where to Look

| I want to... | Look at... |
|---|---|
| Change the PDE, BCs, or mesh | `src/forward_solver.py` (`__init__`, weak form at ~L92) |
| Change the objective or gradient | `src/adjoint_solver.py` + `tao_solver.py:_objgrad` |
| Change the optimizer, bounds, parametrization | `src/tao_solver.py` |
| Accept a new input type for `h` or `q` | `src/domain_coefficient.py:_coefficient_from_user_input` |
| Add a gradient test | `tests/test_grad_*.py`, reuse `tests/_helpers.py` |
| See end-to-end usage | `notebooks/InverseSolve.ipynb` |

## Observations (verified, not yet fixed)

- **Adjoint Dirichlet value.** The adjoint BC should be `λ(y=0)=0`, but the tests and
  `InverseSolve.ipynb` pass `DBC_value=300`. This is currently harmless: with a *single*
  constant-Dirichlet edge and Neumann elsewhere, a constant field lies in the kernel of
  `∇·(h∇·)`, so `λ₃₀₀ = λ₀ + 300` and the gradient — which only sees `∇λ` — is unchanged.
  It would silently break if the Dirichlet data became non-constant or a second Dirichlet
  edge were added. Prefer `DBC_value=0.0` (as `test_grad_forwarddiff.py:78` does).
- **`domain_coefficient.py` tabulated input.** `_parse_tab`'s DataFrame branch indexes by
  integer position (`tab[cols.index("x")]`) instead of by column name, so a real
  `(x|y|value)` frame raises `KeyError`, not the caught `ValueError`. Only the `(N,3)`
  `np.ndarray` path works. Related: the `tab_interpolator` kwarg is dead — it is assigned
  *after* `_build()` has already run and is never read; `RBFInterpolator` is hard-coded
  and the `CloughTocher2DInterpolator` alternative is commented out.
- **TAO requires a `fem.Function` `h`.** `tao_solver.py:74` reads
  `fwd.h.function.function_space`, so a scalar/`fem.Constant` initial guess (`h=4.0`)
  dies with `AttributeError: 'Constant' object has no attribute 'function_space'`. The
  initial guess must be a callable or a tabulated `(N,3)` array. Note also that tabulated
  points are interpolated in *physical* coordinates — `InverseSolve.ipynb` builds its grid
  from integer indices (`np.arange(nmesh)`), which only survives because its initial guess
  is constant and RBF extrapolates a constant exactly.
- **`tao_solver.py:71`**: `h_min=None` is allowed by the type hint but raises `TypeError`
  under `use_logh=True`, because `h_min <= 0.0` is evaluated before the `None` check in
  `_set_tao_bounds_on_logh`.
- **`forward_solver.py:207`**: the `T_obs` assertion message says "Call solve() first";
  it should say `add_noise()`.
- **Branch state (2026-08-12)**: `mf_optimization` has zero commits over `master`; the name
  points at planned multi-fidelity optimization work that has not started. Live uncommitted
  work is `notebooks/EvaluateSolution.ipynb` — a σ×α regularization sweep over the
  `hsol_sigma*_alpha*.npy` grid with a Fourier transfer-function `T(k)` analysis and
  reconstruction-error histograms — plus the new `docs/` tree and the README rewrite.

## Documentation

Sphinx + MyST + `sphinx_rtd_theme`, deployed via Read the Docs (`.readthedocs.yaml`,
`fail_on_warning: true`).

**The docs build never installs the scientific stack.** `docs/conf.py` mocks it through
`autodoc_mock_imports` (dolfinx, petsc4py, ufl, basix, mpi4py, numpy, scipy, pandas,
pyvista, h5py) and puts `src/` — not its parent — on `sys.path`, because the modules are
flat. Consequence to remember: under Sphinx's mocks, `fem.Function` is a *falsy*
`_MockObject`, so any `X or Y` annotation evaluates its right-hand side. That is exactly
what broke the build on `adjoint_solver.py`'s old `T_obs: fem.Function or array - like`
annotation (now `Union[fem.Function, np.ndarray]`).

Class docstrings put equations on an indented line under a `:`-terminated lead-in. reST
reads that as a definition list or block quote, so **each indented equation needs a blank
line before the next unindented paragraph** — otherwise `-W` fails the build. Keep that
spacing when editing docstrings.
