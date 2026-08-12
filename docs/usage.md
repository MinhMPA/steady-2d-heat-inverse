# Usage

Three classes cover the whole workflow, each layered on the previous one:

| Class | Module | Role |
|---|---|---|
| {py:class}`~forward_solver.SteadyHeat2DForwardSolver` | `forward_solver` | mesh, weak form, forward solve, noise, IO, plots |
| {py:class}`~adjoint_solver.SteadyHeat2DAdjointSolver` | `adjoint_solver` | adjoint solve, gradient assembly |
| {py:class}`~tao_solver.SteadyHeat2DTAOSolver` | `tao_solver` | TAO optimization loop |

## Forward solve

```python
from forward_solver import SteadyHeat2DForwardSolver

def h_func(x):
    return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)

fwd = SteadyHeat2DForwardSolver(
    nmesh=128,
    mesh_type="quadrilateral",
    h=h_func,
    q=1.0,
    DBC_value=300.0,
    petsc_opts={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
)
T = fwd.solve()
```

### Constructor options

| Argument | Default | Notes |
|---|---|---|
| `nmesh` | `64` | cells per side of the unit square |
| `mesh_type` | `"quadrilateral"` | or `"triangle"`; anything else raises `ValueError` |
| `h` | `1.0` | thermal conductivity, see [Coefficient inputs](#coefficient-inputs) |
| `q` | `1.0` | heat source, same accepted types as `h` |
| `DBC_value` | `300.0` | $T(y{=}0)$ in Kelvin |
| `petsc_opts` | `None` | merged **over** `{"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}` |
| `tab_interpolator` | `"rbf"` | `"rbf"` or `"ct"`; only affects tabulated `h`/`q` input |

A direct solver is a drop-in alternative when the problem is small — the matrix is SPD, so
Cholesky is the right factorization:

```python
petsc_opts = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}
```

(coefficient-inputs)=
### Coefficient inputs

`h` and `q` are both routed through
{py:class}`domain_coefficient.BaseDomainCoefficient`, which coerces user input into
something UFL can treat as a coefficient:

| Input type | Becomes | Notes |
|---|---|---|
| `int` / `float` | `dolfinx.fem.Constant` | `.constant` attribute set to `True` |
| `dolfinx.fem.Constant` | itself | |
| `dolfinx.fem.Expression` | `fem.Function` | interpolated onto `V` |
| `callable(x)` | `fem.Constant` or `fem.Function` | called on `ufl.SpatialCoordinate`; a callable returning a plain scalar collapses to a `Constant` |
| `numpy.ndarray` of shape `(N, 3)` | `fem.Function` | columns `[x, y, value]`, interpolated with `scipy.interpolate.RBFInterpolator` (cubic kernel, degree-1 polynomial tail) |
| `pandas.DataFrame` with `x`, `y`, `value` columns | `fem.Function` | column names are matched case-insensitively; a missing column raises `ValueError` |

Anything else raises `TypeError`.

:::{note}
Tabulated input is interpolated in **physical** coordinates, so build the grid on
$[0,1]^2$ — not on integer indices.

`tab_interpolator` selects the scheme, on either the coefficient classes or the forward
solver:

| Value | Interpolator | Trade-off |
|---|---|---|
| `"rbf"` (default) | `scipy.interpolate.RBFInterpolator`, cubic kernel, degree-1 tail | extrapolates outside the convex hull; memory ~ $O(N^2)$ |
| `"ct"` | `scipy.interpolate.CloughTocher2DInterpolator`, rescaled | 2D only, no extrapolation (falls back to the mean); memory ~ $O(N\log N)$ |

```python
fwd = SteadyHeat2DForwardSolver(nmesh=128, h=h_table, q=1.0, tab_interpolator="ct")
```
:::

## Adding noise and exporting

```python
T_obs = fwd.add_noise(mu=0.0, sigma=1e-3, seed=0)   # requires solve() first
fwd.export_xdmf("test_data/blackbox_output.xdmf")
```

`export_xdmf` writes the mesh plus `Temperature`, `ObservedTemperature` (if noise was
added), `ThermalConductivity` and `HeatSource` into a paired `.xdmf` / `.h5` file.
Constant coefficients are wrapped into a `fem.Function` first so they are exportable.

Noise is generated on rank 0 and broadcast with `MPI.COMM_WORLD.Bcast`, so every rank
observes the same field — important for reproducibility under `mpirun`.

## Plotting

```python
pl = fwd.plot_output_temperature(zero_point=300.0, noiseless=True)
pl.camera_position = "yx"

fwd.h.plot_input_thermal_conductivity(user_scalar_bar={"fmt": "%.2f"})
fwd.q.plot_input_heat_source()
```

All three delegate to {py:func}`plotting_utils.plot_scalar_mesh`, which renders a
`pyvista.UnstructuredGrid` and **returns `None` on any rank other than 0**. `zero_point`
subtracts a reference temperature so the plot shows $\Delta T = T - T_0$; pass
`zero_point=0.0` for the absolute field.

## Adjoint solve and gradient

```python
from adjoint_solver import SteadyHeat2DAdjointSolver

adj = SteadyHeat2DAdjointSolver(
    fwd, T_obs, sigma=1e-3, alpha=5e-3, DBC_value=0.0,
)
lam = adj.solve()
adj.update_gradient()
g = adj.grad          # PETSc.Vec; raises RuntimeError if update_gradient() was not called
```

| Argument | Default | Notes |
|---|---|---|
| `T_obs` | — | `fem.Function` on the same space, or any array-like of DOF values |
| `sigma` | `1.0` | sensor noise std; stored as `sigma2 = sigma**2` |
| `alpha` | `0.0` | Tikhonov weight; `0.0` disables regularization |
| `DBC_value` | `0.0` | $\lambda(y{=}0)$ — see {doc}`theory` |
| `petsc_opts` | `None` | inherits the forward solver's options when `None` |

The adjoint solver **subclasses** the forward solver and shares its mesh, function space,
boundary DOFs, and — critically — the same `h` object. Mutating `fwd.h.function` is
therefore immediately visible to the adjoint form.

:::{warning}
When `T_obs` is a `fem.Function`, the constructor asserts
`T_obs.function_space == self.V`. Two solver instances built with the same `nmesh` still
own **distinct** meshes and function spaces, so an observation synthesized by a *second*
forward solver will trip that assertion. Pass its DOF values instead — `T_obs.x.array` —
which takes the array-like branch:

```python
truth = SteadyHeat2DForwardSolver(nmesh=128, h=h_func, q=1.0)
truth.solve()
T_obs = truth.add_noise(0.0, 1e-3, seed=0)

fwd = SteadyHeat2DForwardSolver(nmesh=128, h=4.0, q=1.0)
fwd.solve()
adj = SteadyHeat2DAdjointSolver(fwd, T_obs.x.array, sigma=1e-3, alpha=5e-3)
```

The `fem.Function` form is correct when the observation is built on `fwd.V` directly, as
the gradient tests do.
:::

## Optimization

```python
from tao_solver import SteadyHeat2DTAOSolver

def monitor(tao):
    print(f"[TAO] iter={tao.getIterationNumber():3d};  J={tao.getObjectiveValue():.6e}")

tao = SteadyHeat2DTAOSolver(
    fwd, adj,
    h_min=1e-16, h_max=None,
    tao_type="blmvm", ls_algorithm="armijo",
    use_logh=True,
    gatol=1e-5, grtol=1e-5, gttol=1e-6,
    mit=1000,
    monitor=monitor,
    verbose=1,
)
h_sol = tao.solve()      # numpy array of h(x,y) DOF values
```

| Argument | Default | Notes |
|---|---|---|
| `h_min` | `1e-16` | lower bound; must be `> 0` when `use_logh=True` |
| `h_max` | `None` | `None` maps to `PETSc.INFINITY` |
| `tao_type` | `"blmvm"` | `"bncg"` also works; both are bound-constrained, gradient-only |
| `ls_algorithm` | `"armijo"` | e.g. `"more-thuente"` |
| `use_logh` | `True` | optimize in $m = \log h$ |
| `gatol` / `grtol` / `gttol` | `1e-6` / `1e-3` / `1e-3` | gradient absolute / relative / convergence tolerances |
| `mit` | `1000` | maximum iterations |
| `monitor` | `None` | callable receiving the `PETSc.TAO` object each iteration |
| `options` | `False` | when `True`, calls `setFromOptions()` on both TAO and its line search |
| `verbose` | `0` | `0`–`3`; `≥2` prints $h$ and $J$, `3` also prints $\lVert G\rVert$ |

`solve()` returns `numpy.exp(...)` of the TAO solution when `use_logh=True`, so the
returned array is always $h$ — never $\log h$. It is also cached on `tao.solution`.
Convergence can be inspected via `tao.tao.getConvergedReason()`
([reason codes](https://petsc.org/release/manualpages/Tao/TaoConvergedReason/)).

:::{warning}
**The initial `h` must be a `fem.Function`, not a `fem.Constant`.** TAO optimizes the DOF
vector of `fwd.h.function`, so a scalar initial guess (`h=4.0`) — which becomes a
`fem.Constant` with no DOFs, see [Coefficient inputs](#coefficient-inputs) — is rejected
with a `TypeError`. It cannot be promoted automatically: the forward and adjoint UFL forms
capture the coefficient by handle when they are built.

Use a callable, or a tabulated `(N,3)` array for a genuinely flat guess:

```python
import numpy as np
g = np.linspace(0.0, 1.0, nmesh)
xx, yy = np.meshgrid(g, g, indexing="ij")
h_init = np.column_stack([xx.ravel(), yy.ravel(), np.full(nmesh**2, 4.0)])
fwd = SteadyHeat2DForwardSolver(nmesh=nmesh, h=h_init, q=1.0)
```

Note that the tabulated points are interpolated in **physical** coordinates, so build the
grid on $[0,1]^2$ — not on integer indices.
:::

:::{note}
`h_min=None` means "no lower bound" and is accepted when `use_logh=False`, where it maps
to `0.0`. Under `use_logh=True` it raises `ValueError`, since `log(None)` is undefined.
:::

## Running in parallel

Every solver is ghost-aware and works under MPI:

```bash
mpirun -n 4 python script.py
```

Conventions when extending the code:

- after mutating `.x.array`, call `scatter_forward()`;
- after assembling a vector, call `ghostUpdate()`;
- guard prints and plots with `MPI.COMM_WORLD.rank == 0`.

## Notebooks

End-to-end examples live in `notebooks/` (not executed as part of the docs build, since
they require the full DOLFINx stack):

- [`ForwardSolve.ipynb`](https://github.com/MinhMPA/steady-2d-heat-inverse/blob/master/notebooks/ForwardSolve.ipynb)
  — forward solve, noise injection, XDMF export of the synthetic "measured" data.
- [`InverseSolve.ipynb`](https://github.com/MinhMPA/steady-2d-heat-inverse/blob/master/notebooks/InverseSolve.ipynb)
  — full reconstruction from `test_data/blackbox_output.h5`.
- [`EvaluateSolution.ipynb`](https://github.com/MinhMPA/steady-2d-heat-inverse/blob/master/notebooks/EvaluateSolution.ipynb)
  — $\sigma \times \alpha$ regularization sweep, Fourier transfer function $T(k)$, and
  reconstruction-error histograms.
