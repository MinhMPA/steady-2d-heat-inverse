# Defect Fixes and Docs Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the six verified defects found during the documentation audit, cover each with a fast regression test that runs in CI, and re-sync the docs so every warning describing a now-fixed defect is removed.

**Architecture:** All fixes are local edits inside `src/`. Five of the six are input-validation or interpolation bugs reachable from the public constructors, so each gets a fast unit test (seconds, no `gradcheck` marker) in a new `tests/` module. The repo currently has *only* gradient tests and CI runs `pytest -m gradcheck`, so Task 1 also widens CI to run the whole suite — without that, every test this plan adds would be dead weight in CI. The final task re-syncs `docs/` and `CLAUDE.md` and re-verifies the `-W` docs build.

**Tech Stack:** Python 3.11+, DOLFINx/UFL/basix (FEniCSx), petsc4py (KSP + TAO), mpi4py, NumPy/SciPy/pandas, pytest, Sphinx + MyST + sphinx_rtd_theme.

## Global Constraints

- **Import convention:** sources install as *flat top-level modules* (`pyproject.toml` `py-modules`). Always `from forward_solver import ...`, never `from src.forward_solver import ...`.
- **Test imports:** `tests/` is a package (`tests/__init__.py` exists). Import shared helpers relatively: `from ._helpers import ...`. Import the code under test flatly: `from forward_solver import ...`.
- **Formatting:** `black`. Run `black src tests` before each commit.
- **Import banners:** group imports under the existing comment banners (`# type imports`, `# numerical imports`, `# mpi imports`, `# pde imports`, `# dolfinx imports`, `# local imports`). Match this exactly when editing a file.
- **Docstrings:** raw strings (`r"""`) with LaTeX-style math. Every public method documents Parameters and Returns.
- **Docstring reST rule:** an indented equation line under a `:`-terminated lead-in parses as a reST definition list. It **must** be followed by a blank line before the next unindented paragraph, or `sphinx-build -W` fails.
- **MPI hygiene:** after mutating `.x.array`, call `scatter_forward()`; after assembling a vector, `ghostUpdate()`. Print and plot only on `MPI.COMM_WORLD.rank == 0`.
- **Commits:** short imperative subject lines, no prefixes or scopes (e.g. `Fix DataFrame parsing in _parse_tab`). This overrides any `feat:`/`fix:` convention shown in generic examples.
- **New tests must be fast** (< ~5 s each) and must **not** carry `pytest.mark.gradcheck`. Use `nmesh=4` or `nmesh=8`.
- **Do not touch** `notebooks/**` or `test_data/**` — both hold uncommitted user work.

## Out of Scope

These are known and deliberately left alone:

- `notebooks/InverseSolve.ipynb` builds its tabulated initial guess from integer indices (`np.arange(nmesh)`) instead of physical `[0,1]²` coordinates. It survives only because a constant field extrapolates exactly under RBF. Notebooks hold uncommitted work; fixing this is a separate, user-driven change.
- `notebooks/InverseSolve.ipynb` passes `DBC_value=300` to the adjoint solver. Same reason.
- `src/__init__.py`'s relative imports are vestigial under the flat-module layout. Task 5 corrects the misspelled name inside `__all__` but does **not** delete or restructure the file.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tests/test_domain_coefficient.py` | create | fast unit tests for coefficient coercion: DataFrame parsing, interpolator selection |
| `tests/test_solver_validation.py` | create | fast unit tests for constructor validation in the forward and TAO solvers |
| `.github/workflows/run-gradchecks.yml` | modify | run the full pytest suite, not only `-m gradcheck` |
| `src/domain_coefficient.py` | modify | fix DataFrame parsing; fix `tab_interpolator` ordering and wire it up |
| `src/forward_solver.py` | modify | pass `tab_interpolator` through; correct the `T_obs` assertion message |
| `src/tao_solver.py` | modify | reject a `fem.Constant` `h`; reject `h_min=None` under `use_logh` |
| `src/__init__.py` | modify | correct the misspelled class name in `__all__` |
| `tests/test_grad_finitediff.py` | modify | adjoint `DBC_value` → `0.0` |
| `tests/test_grad_taylorexp.py` | modify | adjoint `DBC_value` → `0.0` |
| `docs/usage.md` | modify | drop warnings for fixed defects; document new behavior |
| `CLAUDE.md` | modify | prune fixed entries from Observations |

---

### Task 1: DataFrame coefficient input, with a CI-visible test suite

The `_parse_tab` DataFrame branch indexes by integer position (`tab[cols.index("x")]`)
rather than by column name, so a real `(x|y|value)` frame raises `KeyError: 0`. The
surrounding `except ValueError` never catches it. This task fixes that and, because the
repo has no fast tests today, also makes CI run them.

**Files:**
- Create: `tests/test_domain_coefficient.py`
- Modify: `src/domain_coefficient.py:129-163` (`_parse_tab`)
- Modify: `.github/workflows/run-gradchecks.yml`

**Interfaces:**
- Consumes: `ThermalConductivity(user_input, mesh, V)` from `domain_coefficient` — existing public constructor, unchanged signature in this task.
- Produces: `tests/test_domain_coefficient.py` with module-level helper `_linear_table(n: int = 5) -> np.ndarray` returning an `(n*n, 3)` array of `[x, y, 1.0 + x]` over the unit square, and a pytest fixture `unit_square` returning `(mesh.Mesh, fem.FunctionSpace)`. Task 2 reuses both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_domain_coefficient.py`:

```python
# tests/test_domain_coefficient.py
# numerical imports
import numpy as np
import pandas as pd
import pytest

# mpi imports
from mpi4py import MPI

# dolfinx imports
from dolfinx import mesh, fem

# local imports
from domain_coefficient import ThermalConductivity


@pytest.fixture
def unit_square():
    """A small quadrilateral unit-square mesh and its P1 Lagrange space."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V


def _linear_table(n: int = 5) -> np.ndarray:
    """Tabulate h(x,y) = 1 + x on a regular (n,n) grid over the unit square."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), (1.0 + xx).ravel()])


def test_dataframe_input_matches_ndarray_input(unit_square):
    """A (x|y|value) DataFrame must interpolate identically to the equivalent (N,3) array."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame(
        {"x": table[:, 0], "y": table[:, 1], "value": table[:, 2]}
    )

    from_array = ThermalConductivity(table, domain, V)
    from_frame = ThermalConductivity(frame, domain, V)

    np.testing.assert_allclose(
        from_frame.function.x.array, from_array.function.x.array, rtol=1e-12
    )


def test_dataframe_column_names_are_case_insensitive(unit_square):
    """Column matching lower-cases the header, so (X|Y|Value) is accepted."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame(
        {"X": table[:, 0], "Y": table[:, 1], "Value": table[:, 2]}
    )

    from_array = ThermalConductivity(table, domain, V)
    from_frame = ThermalConductivity(frame, domain, V)

    np.testing.assert_allclose(
        from_frame.function.x.array, from_array.function.x.array, rtol=1e-12
    )


def test_dataframe_missing_column_raises_valueerror(unit_square):
    """A frame without a `value` column must raise ValueError, not KeyError."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame({"x": table[:, 0], "y": table[:, 1], "h": table[:, 2]})

    with pytest.raises(ValueError, match=r"\(x\|y\|value\)"):
        ThermalConductivity(frame, domain, V)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_domain_coefficient.py -v`

Expected: `test_dataframe_input_matches_ndarray_input` and
`test_dataframe_column_names_are_case_insensitive` FAIL with `KeyError: 0`.
`test_dataframe_missing_column_raises_valueerror` FAILS too — it raises `KeyError: 0`
instead of the expected `ValueError`.

- [ ] **Step 3: Fix the DataFrame branch**

In `src/domain_coefficient.py`, replace the `isinstance(tab, pd.DataFrame)` branch of
`_parse_tab`:

```python
        if isinstance(tab, pd.DataFrame):
            cols = [c.lower() for c in tab.columns]
            try:
                x_col = tab.columns[cols.index("x")]
                y_col = tab.columns[cols.index("y")]
                value_col = tab.columns[cols.index("value")]
            except ValueError:
                raise ValueError("pd.DataFrame format must be (x|y|value).")
            pts = np.column_stack([tab[x_col].values, tab[y_col].values])
            vals = tab[value_col].values
```

`cols.index(...)` raises `ValueError` when the name is absent — that is the error the
`except` clause was always meant to catch. Indexing `tab.columns[...]` converts the
position back into the frame's real column label, which is what `tab[...]` expects.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_domain_coefficient.py -v`
Expected: 3 passed.

- [ ] **Step 5: Widen CI to run the whole suite**

The new tests carry no marker, so `pytest -m gradcheck` skips them. In
`.github/workflows/run-gradchecks.yml`, change the workflow name and the final step:

```yaml
name: Run tests
```

```yaml
      - name: Run test suite
        shell: bash -l {0}
        run: pytest -q --maxfail=1 --durations=10
```

Leave the `on:`, `env:`, checkout, micromamba and install steps untouched.

- [ ] **Step 6: Run the full suite locally**

Run: `pytest -q`
Expected: 6 passed (3 gradchecks + 3 new tests).

- [ ] **Step 7: Format and commit**

```bash
black src tests
git add tests/test_domain_coefficient.py src/domain_coefficient.py .github/workflows/run-gradchecks.yml
git commit -m "Fix DataFrame parsing in _parse_tab and run full suite in CI"
```

---

### Task 2: Make `tab_interpolator` selection real

`BaseDomainCoefficient.__init__` assigns `self._tab_interpolator` *after* `self._build()`
has already run, so the attribute is never visible to the code that would use it — and no
code reads it anyway. `RBFInterpolator` is hardcoded and the `CloughTocher2DInterpolator`
alternative sits commented out. The declared default is `"ct"` while actual behavior is
RBF.

This task fixes the assignment order, wires the choice into the interpolation branch, and
**changes the declared default to `"rbf"` to preserve today's behavior**. It also plumbs
the option through `SteadyHeat2DForwardSolver`, since otherwise the keyword remains
unreachable from the public solver API.

**Files:**
- Modify: `src/domain_coefficient.py:34-46` (`__init__`), `src/domain_coefficient.py:93-114` (ndarray branch)
- Modify: `src/forward_solver.py:33-72` (`__init__`)
- Modify: `tests/test_domain_coefficient.py`

**Interfaces:**
- Consumes: `unit_square` fixture and `_linear_table()` from Task 1.
- Produces: `BaseDomainCoefficient(..., *, tab_interpolator: Literal["rbf", "ct"] = "rbf")`, honored at build time and validated eagerly. `SteadyHeat2DForwardSolver(..., tab_interpolator: str = "rbf")` forwards the same value to both `ThermalConductivity` and `HeatSource`. Adds module-level helper `_nonlinear_table(n: int = 4) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

First add the SciPy reference import to the `# numerical imports` banner at the top of
`tests/test_domain_coefficient.py`:

```python
from scipy.interpolate import CloughTocher2DInterpolator
```

Then append to the same file:

```python
def _nonlinear_table(n: int = 4) -> np.ndarray:
    """A coarse, strongly nonlinear table: the two schemes must disagree off its nodes."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack(
        [xx.ravel(), yy.ravel(), (np.sin(4.0 * xx) * np.exp(yy)).ravel()]
    )


def test_both_interpolators_reproduce_a_linear_field(unit_square):
    """h = 1 + x is linear, so RBF (degree-1 tail) and Clough-Tocher both reproduce it."""
    domain, V = unit_square
    table = _linear_table()
    expected = 1.0 + V.tabulate_dof_coordinates()[:, 0]

    for kind in ("rbf", "ct"):
        coeff = ThermalConductivity(table, domain, V, tab_interpolator=kind)
        np.testing.assert_allclose(coeff.function.x.array, expected, atol=1e-8)


def test_tab_interpolator_actually_selects_the_scheme():
    """On a mesh finer than the table, the two schemes must produce different fields.

    A linear field cannot detect this bug -- both schemes reproduce it exactly -- so this
    test deliberately uses a coarse nonlinear table on an 8x8 mesh.
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    table = _nonlinear_table()

    rbf = ThermalConductivity(table, domain, V, tab_interpolator="rbf")
    ct = ThermalConductivity(table, domain, V, tab_interpolator="ct")

    difference = np.abs(rbf.function.x.array - ct.function.x.array).max()
    assert difference > 1e-3, (
        "Both coefficients produced the same field, so tab_interpolator was ignored."
    )

    # And the "ct" field must match SciPy's Clough-Tocher result directly.
    coords = V.tabulate_dof_coordinates()[:, :2]
    expected_ct = CloughTocher2DInterpolator(
        table[:, :2], table[:, 2], fill_value=table[:, 2].mean(), rescale=True
    )(coords)
    np.testing.assert_allclose(ct.function.x.array, expected_ct, atol=1e-12)


def test_tab_interpolator_defaults_to_rbf(unit_square):
    """The default must stay RBF -- that is the behavior every existing result relies on."""
    domain, V = unit_square
    table = _linear_table()

    default = ThermalConductivity(table, domain, V)
    explicit_rbf = ThermalConductivity(table, domain, V, tab_interpolator="rbf")

    assert default._tab_interpolator == "rbf"
    np.testing.assert_allclose(
        default.function.x.array, explicit_rbf.function.x.array, rtol=1e-12
    )


def test_unknown_tab_interpolator_raises_valueerror(unit_square):
    domain, V = unit_square

    with pytest.raises(ValueError, match="Unsupported tab_interpolator"):
        ThermalConductivity(_linear_table(), domain, V, tab_interpolator="linear")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_domain_coefficient.py -v -k "interpolator or linear_field"`

Expected, precisely:
- `test_tab_interpolator_actually_selects_the_scheme` FAILS on
  `assert difference > 1e-3` — today both constructions run RBF, so the two fields are
  bit-identical and `difference` is `0.0`. This is the test that actually pins the bug.
- `test_tab_interpolator_defaults_to_rbf` FAILS on
  `assert default._tab_interpolator == "rbf"` — the current default is `"ct"`.
- `test_unknown_tab_interpolator_raises_valueerror` FAILS — no validation exists, so
  nothing is raised.
- `test_both_interpolators_reproduce_a_linear_field` **passes already**. It is a
  correctness guard for after the fix, not a bug detector: a linear field is reproduced
  exactly by both schemes, so it cannot tell them apart. Do not be misled by it.

- [ ] **Step 3: Fix the assignment order and validate eagerly**

In `src/domain_coefficient.py`, replace `BaseDomainCoefficient.__init__`:

```python
    def __init__(
        self,
        user_input: UserInput,
        mesh: mesh.Mesh,
        V: fem.FunctionSpace,
        *,
        tab_interpolator: Literal["rbf", "ct"] = "rbf",
    ):
        if tab_interpolator not in ("rbf", "ct"):
            raise ValueError(
                f"Unsupported tab_interpolator: {tab_interpolator}. "
                "Supported types: ['rbf','ct']."
            )
        self._user_input = user_input
        self._mesh = mesh
        self._V = V
        # Must precede _build(): the tabulated branch reads it.
        self._tab_interpolator = tab_interpolator
        self.function = self._build()
```

- [ ] **Step 4: Honor the choice in the tabulated branch**

In `src/domain_coefficient.py`, replace the body of the
`isinstance(self._user_input, (np.ndarray, pd.DataFrame))` branch of
`_coefficient_from_user_input`, from the `interp = RBFInterpolator(...)` call up to and
including the preceding comment block:

```python
        if isinstance(self._user_input, (np.ndarray, pd.DataFrame)):
            self.constant = False
            pts, vals = self._parse_tab(self._user_input)
            if self._tab_interpolator == "ct":
                ## piece-wise cubic interpolation, rescale to unit square before interpolating
                ## no extrapolation, values outside of the point convex hull are set to `fill_value`
                ## only works in 2D, but memory ~ O(N log N)
                interp = CloughTocher2DInterpolator(
                    pts, vals, fill_value=vals.mean(), rescale=True
                )
            else:
                ## radial basis function interpolation, cubic kernel with degree-1 polynomial added
                ## more hyperparameters but offers smoothing at sampled data points and extrapolation
                ## works in any dimension, but memory ~ O(N^2), can adjust neighbors to reduce memory usage
                interp = RBFInterpolator(
                    pts, vals, kernel="cubic", neighbors=None, smoothing=0.0, degree=1
                )
            f = fem.Function(self._V)

            def interpolate_func(x):
                points = np.column_stack([x[0], x[1]])
                values = interp(points)
                return values

            f.interpolate(interpolate_func)
            return f
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_domain_coefficient.py -v`
Expected: 7 passed.

- [ ] **Step 6: Plumb the option through the forward solver**

In `src/forward_solver.py`, add the keyword to `SteadyHeat2DForwardSolver.__init__`'s
signature, immediately after `petsc_opts`:

```python
        petsc_opts: dict = None,
        tab_interpolator: str = "rbf",
    ):
```

Add its Parameters entry to the docstring, immediately after the `petsc_opts` line:

```
        tab_interpolator : interpolation scheme for tabulated h/q input, "rbf" or "ct".
```

Then forward it at the two construction sites:

```python
        # Define thermal conductivity and heat source as domain coefficients.
        self.h = ThermalConductivity(
            h, self.mesh, self.V, tab_interpolator=tab_interpolator
        )
        self.q = HeatSource(q, self.mesh, self.V, tab_interpolator=tab_interpolator)
```

- [ ] **Step 7: Verify the passthrough reaches the coefficient**

Append to `tests/test_domain_coefficient.py`:

```python
def test_forward_solver_forwards_tab_interpolator():
    """The forward solver must expose the interpolator choice, not silently drop it."""
    from forward_solver import SteadyHeat2DForwardSolver

    fwd = SteadyHeat2DForwardSolver(
        nmesh=4, h=_linear_table(), q=1.0, tab_interpolator="ct"
    )

    assert fwd.h._tab_interpolator == "ct"
    assert fwd.q._tab_interpolator == "ct"
```

Run: `pytest tests/test_domain_coefficient.py -v`
Expected: 8 passed.

- [ ] **Step 8: Confirm nothing regressed**

Run: `pytest -q`
Expected: 11 passed. The gradchecks must still pass — they use callable `h`, so they never
enter the tabulated branch, but this confirms the changed `__init__` signature broke
nothing.

- [ ] **Step 9: Format and commit**

```bash
black src tests
git add src/domain_coefficient.py src/forward_solver.py tests/test_domain_coefficient.py
git commit -m "Wire up tab_interpolator selection and expose it on the forward solver"
```

---

### Task 3: Reject a constant `h` in the TAO solver

`SteadyHeat2DTAOSolver.__init__` reads `self.fwd.h.function.function_space`, which only
exists on a `fem.Function`. A scalar initial guess (`h=4.0`) becomes a `fem.Constant` with
no DOFs, and the constructor dies with a bare
`AttributeError: 'Constant' object has no attribute 'function_space'`.

The fix is a clear, early error — **not** auto-promotion. Promoting the `Constant` to a
`Function` inside the TAO solver would be too late: the forward and adjoint UFL forms
already captured the `Constant` by handle when they were built, so they would keep using
the old object.

**Files:**
- Create: `tests/test_solver_validation.py`
- Modify: `src/tao_solver.py:64-70` (`__init__`, before the `use_logh` branch)

**Interfaces:**
- Consumes: `SteadyHeat2DForwardSolver`, `SteadyHeat2DAdjointSolver`, `SteadyHeat2DTAOSolver` — existing constructors.
- Produces: `tests/test_solver_validation.py` with a module-level helper `_solver_pair(h)` returning a `(forward, adjoint)` tuple wired for a tiny `nmesh=4` problem. Task 4 reuses it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_solver_validation.py`:

```python
# tests/test_solver_validation.py
# numerical imports
import numpy as np
import pytest

# dolfinx imports
from dolfinx import fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver


def _solver_pair(h):
    """Build a tiny solved forward/adjoint pair for the given thermal conductivity `h`."""
    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=h, q=1.0, DBC_value=300.0)
    fwd.solve()
    T_obs = fem.Function(fwd.V)
    T_obs.x.array[:] = fwd.T.x.array
    T_obs.x.scatter_forward()
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs, sigma=1.0, alpha=0.0, DBC_value=0.0
    )
    adj.solve()
    return fwd, adj


def test_constant_h_is_rejected_with_actionable_error():
    """A scalar h has no DOF vector for TAO to optimize -- say so, do not AttributeError."""
    fwd, adj = _solver_pair(4.0)

    with pytest.raises(TypeError, match="must be a fem.Function"):
        SteadyHeat2DTAOSolver(fwd, adj)


def test_callable_h_is_accepted():
    """A spatially-varying h yields a fem.Function and must construct cleanly."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    tao = SteadyHeat2DTAOSolver(fwd, adj)

    assert isinstance(fwd.h.function, fem.Function)
    assert tao.use_logh is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_solver_validation.py -v`

Expected: `test_constant_h_is_rejected_with_actionable_error` FAILS — it raises
`AttributeError: 'Constant' object has no attribute 'function_space'` instead of the
expected `TypeError`. `test_callable_h_is_accepted` already PASSES.

- [ ] **Step 3: Add the guard**

In `src/tao_solver.py`, in `__init__`, insert immediately after `self.verbose = verbose`
and before `if self.use_logh:`:

```python
        # TAO optimizes the DOF vector of h, so h must carry DOFs. A scalar or
        # fem.Constant h cannot be promoted here: the forward and adjoint UFL forms
        # already reference the original object by handle.
        if not isinstance(self.fwd.h.function, fem.Function):
            raise TypeError(
                "Thermal conductivity h must be a fem.Function to be optimized: TAO "
                "optimizes its DOF vector, and a scalar or fem.Constant h has no DOFs. "
                "Pass a callable or a tabulated (N,3) array as `h` to the forward solver."
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_solver_validation.py -v`
Expected: 2 passed.

- [ ] **Step 5: Format and commit**

```bash
black src tests
git add src/tao_solver.py tests/test_solver_validation.py
git commit -m "Reject constant thermal conductivity in the TAO solver"
```

---

### Task 4: Validate `h_min=None` under the log parametrization

`h_min` is annotated `float | None`, but with `use_logh=True` the constructor evaluates
`h_min <= 0.0` before anything checks for `None`, raising
`TypeError: '<=' not supported between instances of 'NoneType' and 'float'`. `None` is
legitimate when `use_logh=False` (it maps to a `0.0` lower bound), so the annotation stays
— only the guard changes.

**Files:**
- Modify: `src/tao_solver.py:71-73` (the `use_logh` guard)
- Modify: `tests/test_solver_validation.py`

**Interfaces:**
- Consumes: `_solver_pair(h)` from Task 3.
- Produces: no new symbols.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_solver_validation.py`:

```python
def test_h_min_none_under_logh_raises_valueerror():
    """log(None) is meaningless -- reject it with the same ValueError as h_min <= 0."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    with pytest.raises(ValueError, match="h_min must be positive"):
        SteadyHeat2DTAOSolver(fwd, adj, h_min=None, use_logh=True)


def test_h_min_none_is_allowed_without_logh():
    """Optimizing directly in h, a None lower bound simply means 0.0."""
    fwd, adj = _solver_pair(lambda x: 2.0 + 3.0 * x[0] ** 2)

    tao = SteadyHeat2DTAOSolver(fwd, adj, h_min=None, use_logh=False)

    assert tao.lb.min()[1] == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_solver_validation.py -v -k h_min`

Expected: `test_h_min_none_under_logh_raises_valueerror` FAILS — it raises `TypeError`,
not `ValueError`. `test_h_min_none_is_allowed_without_logh` already PASSES.

- [ ] **Step 3: Check for `None` before comparing**

In `src/tao_solver.py`, replace the guard at the top of the `use_logh` branch:

```python
        if self.use_logh:
            if h_min is None or h_min <= 0.0:
                raise ValueError("h_min must be positive to define log(h_min).")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_solver_validation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Format and commit**

```bash
black src tests
git add src/tao_solver.py tests/test_solver_validation.py
git commit -m "Reject h_min=None under the log parametrization"
```

---

### Task 5: Correct two misleading strings

Two pure-text defects, neither of which changes control flow:

1. `forward_solver.py` asserts on `T_obs` with the message `"No solution available. Call
   solve() first."` — but `solve()` never creates `T_obs`. The caller needs `add_noise()`.
2. `src/__init__.py`'s `__all__` misspells `SteadyHeat2DForwardSolver` as
   `SteadyHeatForwardSolver2D`.

The assertion message is user-visible behavior and gets a test. `__all__` is not: the file
is vestigial under the flat-module layout and is never imported, so a test would assert
nothing meaningful. Correct it without one.

**Files:**
- Modify: `src/forward_solver.py:207`
- Modify: `src/__init__.py:6`
- Modify: `tests/test_solver_validation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no new symbols.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_solver_validation.py`:

```python
def test_plotting_noisy_field_without_noise_points_at_add_noise():
    """The assertion must name the method that actually produces T_obs."""
    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0 + x[0], q=1.0)
    fwd.solve()

    with pytest.raises(AssertionError, match="add_noise"):
        fwd.plot_output_temperature(noiseless=False)
```

The assertion fires before any PyVista call, so no plot window opens.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_solver_validation.py::test_plotting_noisy_field_without_noise_points_at_add_noise -v`
Expected: FAIL — the raised `AssertionError` reads `"No solution available. Call solve()
first."`, which does not match `add_noise`.

- [ ] **Step 3: Fix the message**

In `src/forward_solver.py`, in the `else:` branch of `plot_output_temperature`:

```python
            assert hasattr(self, "T_obs"), (
                "No noisy observation available. Call add_noise() first."
            )
```

Leave the `if noiseless:` branch's `"No solution available. Call solve() first."`
untouched — it is correct.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_solver_validation.py -v`
Expected: 5 passed.

- [ ] **Step 5: Correct the name in `__all__`**

In `src/__init__.py`:

```python
__all__ = [
    "SteadyHeat2DForwardSolver",
    "SteadyHeat2DAdjointSolver",
    "SteadyHeat2DTAOSolver",
]
```

Change nothing else in the file — the relative imports stay as they are.

- [ ] **Step 6: Confirm the whole suite is green**

Run: `pytest -q`
Expected: 16 passed.

- [ ] **Step 7: Format and commit**

```bash
black src tests
git add src/forward_solver.py src/__init__.py tests/test_solver_validation.py
git commit -m "Correct the T_obs assertion message and the __all__ class name"
```

---

### Task 6: Use the correct adjoint Dirichlet value in the gradient tests

The adjoint BC should be `λ(y=0) = 0`, and `DBC_value` already defaults to `0.0`. Two of
the three gradient tests nonetheless pass `DBC_value=T_bottom` (300). That is currently
harmless — with a single constant-Dirichlet edge and Neumann elsewhere, a constant field
lies in the kernel of `∇·(h∇·)`, so `λ₃₀₀ = λ₀ + 300` and the gradient, which only sees
`∇λ`, is unchanged. It would break silently the moment the Dirichlet data became
non-constant or a second Dirichlet edge appeared.

`test_grad_forwarddiff.py` already passes `0.0`. This aligns the other two.

**Files:**
- Modify: `tests/test_grad_finitediff.py:59`
- Modify: `tests/test_grad_taylorexp.py:80`

**Interfaces:**
- Consumes: nothing new.
- Produces: no new symbols.

- [ ] **Step 1: Record the current gradient values**

Run: `pytest -q -m gradcheck`
Expected: 3 passed. These must still pass unchanged after the edit — that is the whole
verification for this task.

- [ ] **Step 2: Change the finite-difference test**

In `tests/test_grad_finitediff.py`, in the `SteadyHeat2DAdjointSolver(...)` call only:

```python
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=0.0
    )
```

Leave both `SteadyHeat2DForwardSolver(...)` calls at `DBC_value=T_bottom` — 300 K is the
physical forward boundary condition and is correct there.

- [ ] **Step 3: Change the Taylor-remainder test**

In `tests/test_grad_taylorexp.py`, in the `SteadyHeat2DAdjointSolver(...)` call only:

```python
        adj = SteadyHeat2DAdjointSolver(
            fwd, T_obs, sigma=noise_sigma, alpha=reg_alpha, DBC_value=0.0
        )
```

Again, leave the forward-solver calls untouched.

- [ ] **Step 4: Verify the gradients are unchanged**

Run: `pytest -q -m gradcheck`
Expected: 3 passed. The Taylor-remainder rate must still come out ≈ 2. If any test now
fails, the kernel argument above does not hold and the change must be reverted and
investigated rather than forced through.

- [ ] **Step 5: Commit**

```bash
black tests
git add tests/test_grad_finitediff.py tests/test_grad_taylorexp.py
git commit -m "Use the correct adjoint Dirichlet value in the gradient tests"
```

---

### Task 7: Re-sync the documentation

Five of the docs' warning blocks describe defects that Tasks 1–5 just fixed. Leaving them
in place would be worse than having never written them. This task removes or rewrites
each, records the new behavior, and re-verifies the `-W` build.

**Files:**
- Modify: `docs/usage.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the behavior established by Tasks 1–5.
- Produces: no new symbols.

- [ ] **Step 1: Update the coefficient-input table**

In `docs/usage.md`, in the "Coefficient inputs" table, add a `pandas.DataFrame` row
immediately after the `numpy.ndarray` row:

```markdown
| `pandas.DataFrame` with `x`, `y`, `value` columns | `fem.Function` | column names are matched case-insensitively; a missing column raises `ValueError` |
```

- [ ] **Step 2: Replace the DataFrame warning with the interpolator note**

In `docs/usage.md`, delete this block entirely:

```markdown
:::{warning}
A `pandas.DataFrame` input is nominally accepted but the parsing branch in
`BaseDomainCoefficient._parse_tab` indexes the frame by integer position rather than by
column name, so it raises `KeyError` for a real `(x|y|value)` frame. Use the `(N, 3)`
ndarray form. Likewise, the `tab_interpolator` keyword is currently inert — it is assigned
*after* `_build()` has already run and is never read; tabulated input always goes through
`RBFInterpolator`.
:::
```

and put this in its place:

```markdown
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
```

- [ ] **Step 3: Add `tab_interpolator` to the constructor-options table**

In `docs/usage.md`, in the forward-solver "Constructor options" table, append a row after
`petsc_opts`:

```markdown
| `tab_interpolator` | `"rbf"` | `"rbf"` or `"ct"`; only affects tabulated `h`/`q` input |
```

- [ ] **Step 4: Rewrite the TAO constant-`h` warning**

In `docs/usage.md`, replace the body of the warning that begins **"The initial `h` must be
a `fem.Function`, not a `fem.Constant`."** — keep the heading sentence, and replace the
sentence describing the failure:

```markdown
**The initial `h` must be a `fem.Function`, not a `fem.Constant`.** TAO optimizes the DOF
vector of `fwd.h.function`, so a scalar initial guess (`h=4.0`) — which becomes a
`fem.Constant` with no DOFs, see [Coefficient inputs](#coefficient-inputs) — is rejected
with a `TypeError`. It cannot be promoted automatically: the forward and adjoint UFL forms
capture the coefficient by handle when they are built.
```

Leave the rest of that warning — the tabulated-grid example and the physical-coordinates
note — exactly as it is.

- [ ] **Step 5: Update the `h_min` note**

In `docs/usage.md`, replace the `h_min=None` note:

```markdown
:::{note}
`h_min=None` means "no lower bound" and is accepted when `use_logh=False`, where it maps
to `0.0`. Under `use_logh=True` it raises `ValueError`, since `log(None)` is undefined.
:::
```

- [ ] **Step 6: Prune the fixed entries from CLAUDE.md**

In `CLAUDE.md`, under `## Observations (verified, not yet fixed)`, delete these four
bullets in full, since Tasks 1–5 fixed them:

- the `domain_coefficient.py` tabulated-input bullet,
- the "TAO requires a `fem.Function` `h`" bullet,
- the `tao_solver.py:71` bullet,
- the `forward_solver.py:207` bullet.

Then rewrite the "Adjoint Dirichlet value" bullet's closing sentence, because Task 6
changed the tests:

```markdown
  edge were added. The gradient tests now all pass `DBC_value=0.0`; `InverseSolve.ipynb`
  still passes `300`.
```

Finally, add one bullet recording what stays broken:

```markdown
- **`InverseSolve.ipynb` tabulated grid.** It builds the initial guess from integer
  indices (`np.arange(nmesh)`) rather than physical `[0,1]²` coordinates. This survives
  only because the guess is constant and RBF extrapolates a constant exactly; any
  spatially-varying tabulated guess built this way would be silently wrong.
```

- [ ] **Step 7: Update the branch-state note**

In `CLAUDE.md`, replace the "Branch state" bullet:

```markdown
- **Branch state (2026-08-12)**: `mf_optimization` has zero commits over `master`; the name
  points at planned multi-fidelity optimization work that has not started. Live uncommitted
  work is `notebooks/EvaluateSolution.ipynb` — a σ×α regularization sweep over the
  `hsol_sigma*_alpha*.npy` grid with a Fourier transfer-function `T(k)` analysis and
  reconstruction-error histograms.
```

- [ ] **Step 8: Rebuild the docs**

Run: `make -C docs html`
Expected: `build succeeded.` with zero warnings. The Makefile passes `-W --keep-going`, so
any warning fails the build.

- [ ] **Step 9: Run the full suite one last time**

Run: `pytest -q`
Expected: 16 passed.

- [ ] **Step 10: Verify the README quickstart still executes**

The quickstart is the most-read code in the repo and none of the tests cover it. Run it
verbatim at reduced resolution:

```bash
python - <<'EOF'
import sys, re
sys.path.insert(0, "src")
src = re.search(r"```python\n(.*?)```", open("README.md").read(), re.S).group(1)
exec(src.replace("nmesh=128", "nmesh=16").replace("verbose=1", "verbose=0, gatol=1e-3, grtol=1e-3, mit=20"))
print("README quickstart OK -> h range [%.3f, %.3f]" % (h_sol.min(), h_sol.max()))
EOF
```

Expected: `README quickstart OK -> h range [...]`, with the range roughly spanning
`[0.95, 7.03]`.

- [ ] **Step 11: Commit**

```bash
git add docs/usage.md CLAUDE.md
git commit -m "Sync documentation with the defect fixes"
```
