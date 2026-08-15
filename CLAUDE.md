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
  positivity and applies the chain rule `dJ/dm = h · dJ/dh` (`tao_solver.py:_objgrad`).

## Tech Stack

| Layer | Technology |
|---|---|
| FEM | DOLFINx + UFL + basix (FEniCSx) |
| Linear algebra / optimizer | PETSc (KSP: `cg` + `hypre`) and TAO (`blmvm`, `bncg`) via petsc4py |
| Parallelism | MPI via mpi4py (single-program, ghost-aware) |
| Interpolation | SciPy `RBFInterpolator` / `CloughTocher2DInterpolator` |
| Visualization / IO | PyVista (**optional `[plot]` extra**, not in `environment.yml`), XDMF+HDF5, h5py |
| Env / build | conda-forge (`environment.yml`, fully pinned) + setuptools editable install |
| Tests | pytest; slow gradient tests carry a `gradcheck` marker, fast validation tests do not |

**The environment is pinned to DOLFINx 0.11** (`fenics-ufl 2026.1`, `petsc4py 3.25`,
python 3.12). Two consequences that bite if forgotten:

- `LinearProblem` requires a **mandatory `petsc_options_prefix`**. Three call sites use
  distinct prefixes — `s2dhi_fwd_`, `s2dhi_adj_`, `s2dhi_tangent_`. A collision would
  silently apply one solver's PETSc options to another.
- `element.interpolation_points` is a **property**, not a method. Calling it raises
  `TypeError: 'numpy.ndarray' object is not callable`.

**`pyvista`/`vtk` cannot co-exist with dolfinx 0.11** (libboost conflict), so they are
deliberately absent from the core environment and live in the `[plot]` extra.
`plotting_utils` therefore imports pyvista **lazily**, inside `plot_scalar_mesh()`, and
raises a wrapped `ImportError` naming the extra. Keep it lazy: a module-scope import makes
a *plotting* dependency able to render the entire library unimportable, which is exactly
how CI broke.

## Layout

```
src/                       flat top-level modules (see import gotcha below)
  forward_solver.py        SteadyHeat2DForwardSolver — mesh, weak form, solve, noise, XDMF export, plots
  adjoint_solver.py        SteadyHeat2DAdjointSolver — adjoint weak form, solve, assemble_gradient
  tao_solver.py            SteadyHeat2DTAOSolver — TAO wrapper, bounds, log(h) reparam, _objgrad callback
  domain_coefficient.py    BaseDomainCoefficient (ABC) + ThermalConductivity / HeatSource
  plotting_utils.py        plot_scalar_mesh() — PyVista rendering (rank-0 only)
tests/                     3 gradient verifications + 2 fast validation modules + private helpers
notebooks/                 ForwardSolve, InverseSolve, EvaluateSolution
test_data/                 blackbox_output.{xdmf,h5} — synthetic "measured" data
docs/                      Sphinx source (MyST + autodoc); `.readthedocs.yaml` at repo root
```

**Import gotcha (most common mistake):** `pyproject.toml` installs the sources as *flat
top-level modules* via `py-modules`, so it is `from forward_solver import ...` — never
`from src.forward_solver` and never a `steady_2d_heat_inverse` package.
`src/__init__.py` is vestigial: its relative imports are never exercised.

## Data Flow

`h₀` → `ThermalConductivity` coerces input (scalar / callable / `fem.Expression` /
tabulated `(N,3)` array or DataFrame) into a `fem.Constant` or interpolated `fem.Function`
→ forward `LinearProblem` solves for `T` → adjoint solves for `λ` → `assemble_gradient()`
returns `dJ/dh` as a `PETSc.Vec` → TAO's `_objgrad` callback returns `(J, G)` → TAO proposes
a new `h` → `_update_h()` writes it back into the *shared* `fwd.h.function`, which both the
forward and adjoint UFL forms reference by handle. Loop until converged.

## Commands

```bash
mamba env create -n s2dhi-011 -f environment.yml   # mamba: a conda solve risks timing out
conda activate s2dhi-011
pip install -e ".[dev]"                            # add ,plot only if you need figures

pytest -q                    # full suite (23 tests)
pytest -m gradcheck          # adjoint gradient verification only (real FEM solves)
mpirun -n 4 python script.py # MPI-parallel run — see the open risk below

python scripts/make_fixture.py --seed 0 --sigma 1e-3 --nmesh 128  # regenerate test_data/

pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html   # docs build; no conda stack needed
```

The observation fixture is **reproducible**: `test_data/blackbox_output.meta.json` records
the seed and a `T_obs_sha256`, and re-running the generator reproduces it exactly. Never
regenerate it without a seed — an unseeded draw silently changes the inverse problem's
target and makes stored results incomparable.

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
  `pytestmark`; `tests/test_domain_coefficient.py` and `tests/test_solver_validation.py`
  carry no marker. Markers are registered in `pyproject.toml`.

## Testing Strategy

Gradient correctness is verified three independent ways — keep all three passing when
touching `adjoint_solver.py` or the objective:

1. `test_grad_finitediff.py` — adjoint directional derivative vs. central finite difference.
2. `test_grad_forwarddiff.py` — vs. a tangent-linear (JVP) solve from `_tangent_solver.py`.
3. `test_grad_taylorexp.py` — Taylor-remainder convergence rate must be ≈ 2.

CI: `.github/workflows/install-and-import.yml` (env + editable install + import smoke test,
every push) and `run-gradchecks.yml` (micromamba + `pytest -q --maxfail=1 --durations=10`,
the full 16-test suite, on `src/**`, `tests/**`, `pyproject.toml`, `environment.yml`
changes).

## Where to Look

| I want to... | Look at... |
|---|---|
| Change the PDE, BCs, or mesh | `src/forward_solver.py` (weak form in `SteadyHeat2DForwardSolver.__init__`) |
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
  edge were added. The gradient tests now all pass `DBC_value=0.0`; `InverseSolve.ipynb`
  still passes `300`.
- **`InverseSolve.ipynb` tabulated grid.** It builds the initial guess from integer
  indices (`np.arange(nmesh)`) rather than physical `[0,1]²` coordinates. This survives
  only because the guess is constant and RBF extrapolates a constant exactly; any
  spatially-varying tabulated guess built this way would be silently wrong.
- **`mpirun` + TAO may stall.** A `mpirun -n 2` run of a full TAO solve hung past 7.5
  minutes with no output and had to be killed, although trivial MPI works fine in the
  pinned environment (both ranks print, exit 0). Unresolved. This matters because the
  README documents `mpirun -n 4` and Tier 3 makes a 2-rank check its exit gate —
  investigate before relying on multi-rank runs.
- **`src/adjoint_solver.py:60`** has a non-raw docstring containing `\lambda`, which emits
  a `SyntaxWarning` today and becomes a `SyntaxError` on a future CPython. It is the only
  such instance in `src/` and `tests/` (verified by compiling every file with warnings
  escalated). Left alone deliberately: the Tier 3 plan rewrites that file wholesale.

## Project state (2026-08-15)

On `master`, synced with `origin`, **CI green** (`install-and-import`, `gradcheck`, and a
new `Build documentation` job). 23 tests pass.

**Tier 1 of a three-tier refactor is complete.** The plans live in
`docs/superpowers/plans/2026-08-13-tier{1,2,3}-*.md` and are meant to run in order:

- **Tier 1 (done)** — pinned DOLFINx 0.11, lazy pyvista, unconverged solves raise, optimum
  re-synced to the shared `h` after `solve()`, deterministic seeded fixture, CI green.
- **Tier 2 (next)** — structural: a `Discretization` value object, composition replacing
  the adjoint's inheritance of the forward solver, index-map DOF counts. Deliberately
  **behaviour-preserving**; its exit gate is that the directional derivative does not
  change in any digit.
- **Tier 3** — the payoff: a fixed-sensor observation operator `B`, an explicit
  `Σ = σ²I` likelihood, and a pure MPI-reduced `value_and_grad` that both TAO and a
  BlackJAX `custom_vjp` bridge can consume. Unblocks multi-fidelity and HMC.

The tier ordering is load-bearing: Tier 3 rewrites `adjoint_solver.__init__`, and so does
Tier 2's composition work, so the structural change must land first or one redoes the
other.

Three known problems the plans have **not** yet addressed: the objective is not MPI-reduced
(`fem.assemble_scalar` is rank-local), the misfit is mass-matrix weighted `rᵀMr` while
`add_noise` generates i.i.d. per-DOF noise (so `exp(-J)` is not the posterior), and
`add_noise` broadcasts a per-rank-sized buffer. All three are Tier 3's to fix.

## Documentation

Sphinx + MyST + `sphinx_rtd_theme`, published at
**https://steady-2d-heat-inverse.readthedocs.io/** and rebuilt by Read the Docs on every
push to `master` (`.readthedocs.yaml`, `fail_on_warning: true`).

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
