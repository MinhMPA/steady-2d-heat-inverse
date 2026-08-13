# Tier 2: Structural Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the forward/adjoint inheritance with explicit composition, remove the P1-only DOF-count assumption, and correct the remaining API-contract defects — so that Tier 3's objective rework lands on the final class structure and never has to be redone.

**Architecture:** A new `Discretization` value object owns the mesh, function space, and boundary DOFs. Forward and adjoint solvers each *hold* one instead of the adjoint *inheriting* from the forward solver. This tier is deliberately **behavior-preserving**: every existing test must pass with numerically identical results. That property is the verification.

**Tech Stack:** DOLFINx 0.11, PETSc/TAO, pytest.

## Prerequisite

**Tier 1 must be complete and its verification round green.** This tier assumes a pinned DOLFINx 0.11 environment, `petsc_options_prefix` on both solvers, lazy `pyvista`, and green CI.

## Why this tier precedes the objective rework

Tier 3 rewrites `SteadyHeat2DAdjointSolver.__init__` (observations become a sensor-space vector) and its right-hand side (a `Bᵀ`-driven point source). The composition refactor *also* rewrites `__init__`. Running the objective work first would force this tier to edit code Tier 3 had just written. Structure first, behavior second.

## Global Constraints

- **Import convention:** flat top-level modules. Always `from forward_solver import ...`, never `from src.forward_solver import ...`.
- **Test imports:** `tests/` is a package. Import shared helpers relatively (`from ._helpers import ...`); import code under test flatly.
- **Formatting:** `black`. Run `black src tests scripts` before each commit.
- **Import banners:** group imports under the existing comment banners (`# type imports`, `# numerical imports`, `# mpi imports`, `# pde imports`, `# dolfinx imports`, `# local imports`).
- **Docstrings:** raw strings (`r"""`) with LaTeX-style math; document Parameters and Returns.
- **Docstring reST rule:** an indented equation under a `:`-terminated lead-in must be followed by a blank line before the next unindented paragraph, or `sphinx-build -W` fails.
- **MPI hygiene:** after mutating `.x.array`, call `scatter_forward()`; after assembling a vector, `ghostUpdate()`. Print and plot only on rank 0.
- **Commits:** short imperative subject lines, no `feat:`/`fix:` prefixes.
- **Behavior preservation is the contract.** No task in this tier may change a computed number. If a gradient check shifts, stop and investigate rather than adjusting tolerances.
- **Do not touch** `notebooks/**` beyond what a task names.

## Out of Scope (owned by Tier 3)

| Item | Reason |
|---|---|
| MPI reduction of `J` | The new objective module is reduced by construction |
| `add_noise` correctness | Replaced wholesale by the sensor observation model |
| Form caching in `_objgrad` | The objective moves to a new module |
| The misfit's weighting | The likelihood decision belongs to Tier 3 |

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/discretization.py` | create | `Discretization` — mesh, function space, boundary DOFs, DOF counts |
| `src/forward_solver.py` | modify | hold a `Discretization`; use `index_map` DOF counts; fix the `h` callable docstring |
| `src/adjoint_solver.py` | modify | compose rather than inherit; validate `T_obs` with `ValueError` |
| `src/domain_coefficient.py` | modify | use `index_map` DOF counts in the plot helpers |
| `tests/_tangent_solver.py` | modify | compose rather than duplicate the attribute copying |
| `tests/test_discretization.py` | create | DOF-count and boundary-DOF tests, including a P2 space |
| `tests/test_solver_structure.py` | create | composition contract; `T_obs` validation |

---

### Task 1: Extract a `Discretization` value object

The mesh, function space, and bottom-boundary DOFs are currently created in `SteadyHeat2DForwardSolver.__init__` and hand-copied into the adjoint solver (`adjoint_solver.py`) and again into `tests/_tangent_solver.py`. Three copies of the same five lines.

Extracting them into one object is the precondition for composition, and it gives a single place to define DOF counts correctly.

**Files:**
- Create: `src/discretization.py`
- Create: `tests/test_discretization.py`
- Modify: `pyproject.toml` (register the new flat module)

**Interfaces:**
- Consumes: nothing.
- Produces: `Discretization(nmesh, mesh_type, degree=1)` exposing `.mesh`, `.V`, `.bottom_dofs`, `.n_dofs_local`, `.n_dofs_owned`, `.dof_coordinates`. Tasks 2–4 and all of Tier 3 use this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_discretization.py`:

```python
# tests/test_discretization.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# local imports
from discretization import Discretization


def test_dof_counts_match_the_function_space():
    """n_dofs_owned must come from the index map, not from mesh geometry."""
    d = Discretization(nmesh=8, mesh_type="quadrilateral")

    index_map = d.V.dofmap.index_map
    assert d.n_dofs_owned == index_map.size_local
    assert d.n_dofs_local == index_map.size_local + index_map.num_ghosts


def test_p2_dof_count_differs_from_geometry_node_count():
    """The whole point of using the index map: P2 has more dofs than mesh nodes.

    `mesh.geometry.x.shape[0]` silently equals the dof count only for P1.
    """
    d1 = Discretization(nmesh=8, mesh_type="quadrilateral", degree=1)
    d2 = Discretization(nmesh=8, mesh_type="quadrilateral", degree=2)

    n_geometry_nodes = d2.mesh.geometry.x.shape[0]

    assert d1.n_dofs_local == d1.mesh.geometry.x.shape[0]
    assert d2.n_dofs_local > n_geometry_nodes


def test_bottom_dofs_lie_on_y_equals_zero():
    d = Discretization(nmesh=8, mesh_type="quadrilateral")
    coords = d.dof_coordinates[d.bottom_dofs]

    assert coords.shape[0] > 0
    np.testing.assert_allclose(coords[:, 1], 0.0, atol=1e-14)


def test_unsupported_mesh_type_raises():
    with pytest.raises(ValueError, match="Unsupported mesh type"):
        Discretization(nmesh=4, mesh_type="hexahedron")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_discretization.py -v
```

Expected: all four FAIL with `ModuleNotFoundError: No module named 'discretization'`.

- [ ] **Step 3: Write the module**

Create `src/discretization.py`:

```python
# type imports
from typing import Literal

# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# dolfinx imports
from dolfinx import fem, mesh as dmesh


class Discretization:
    r"""
    The finite-element discretization of the unit square :math:`[0,1]^2`.

    Owns the mesh, the Lagrange function space, and the Dirichlet boundary DOFs.
    Forward, adjoint, and tangent-linear solvers all hold one of these rather
    than each rebuilding (or copying) the same objects.

    Parameters
    ----------
    nmesh      : number of cells per side of the unit square.
    mesh_type  : "quadrilateral" or "triangle".
    degree     : Lagrange element degree.
    """

    def __init__(
        self,
        nmesh: int = 64,
        mesh_type: Literal["quadrilateral", "triangle"] = "quadrilateral",
        degree: int = 1,
    ):
        if mesh_type not in ("quadrilateral", "triangle"):
            raise ValueError(
                f"Unsupported mesh type: {mesh_type}. "
                "Supported types: ['quadrilateral','triangle']."
            )
        cell_type = (
            dmesh.CellType.quadrilateral
            if mesh_type == "quadrilateral"
            else dmesh.CellType.triangle
        )
        self.mesh = dmesh.create_unit_square(MPI.COMM_WORLD, nmesh, nmesh, cell_type)
        self.V = fem.functionspace(self.mesh, ("Lagrange", degree))
        self.degree = degree

        # Dirichlet boundary: the bottom wall y = 0.
        facets = dmesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0)
        )
        self.bottom_dofs = fem.locate_dofs_topological(
            self.V, self.mesh.topology.dim - 1, facets
        )

    @property
    def n_dofs_owned(self) -> int:
        """DOFs owned by this MPI rank, excluding ghosts."""
        return self.V.dofmap.index_map.size_local

    @property
    def n_dofs_local(self) -> int:
        """DOFs addressable on this rank, i.e. owned plus ghosts.

        This is the length of ``Function.x.array``.
        """
        index_map = self.V.dofmap.index_map
        return index_map.size_local + index_map.num_ghosts

    @property
    def dof_coordinates(self) -> np.ndarray:
        """(n_dofs_local, 3) physical coordinates of the degrees of freedom."""
        return self.V.tabulate_dof_coordinates()
```

- [ ] **Step 4: Register the module**

In `pyproject.toml`, add `"discretization"` to `py-modules`:

```toml
py-modules = [
  "forward_solver",
  "adjoint_solver",
  "tao_solver",
  "domain_coefficient",
  "discretization",
  "plotting_utils"
]
```

Reinstall so the new module is importable:

```bash
pip install -e . -q
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_discretization.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/discretization.py tests/test_discretization.py pyproject.toml
git commit -m "Extract a Discretization value object"
```

---

### Task 2: Hold a `Discretization` in the forward solver

The forward solver keeps its public API exactly — `nmesh`, `mesh_type`, `self.mesh`, `self.V`, `self.bottom_dofs` all remain — but delegates their construction. This keeps every existing caller and notebook working while giving the adjoint something to share.

**Files:**
- Modify: `src/forward_solver.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `Discretization` from Task 1.
- Produces: `SteadyHeat2DForwardSolver.disc` (a `Discretization`). Task 3's adjoint solver and all of Tier 3 consume it. `self.mesh`, `self.V`, `self.bottom_dofs` remain as delegating attributes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke.py`:

```python
def test_forward_solver_exposes_a_discretization():
    """The forward solver holds a Discretization the adjoint can share."""
    sys.path.insert(0, "src")
    from forward_solver import SteadyHeat2DForwardSolver
    from discretization import Discretization

    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=1.0, q=1.0)

    assert isinstance(fwd.disc, Discretization)
    # Legacy attributes must still resolve to the same objects.
    assert fwd.mesh is fwd.disc.mesh
    assert fwd.V is fwd.disc.V
    assert fwd.bottom_dofs is fwd.disc.bottom_dofs
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_smoke.py::test_forward_solver_exposes_a_discretization -v
```

Expected: FAIL with `AttributeError: 'SteadyHeat2DForwardSolver' object has no attribute 'disc'`.

- [ ] **Step 3: Delegate construction**

In `src/forward_solver.py`, add to the `# local imports` banner:

```python
from discretization import Discretization
```

Then replace the mesh/space/boundary construction block — everything from the `if mesh_type not in [...]` check through the `self.bcs = [...]` assignment — with:

```python
        # Discretization: mesh, function space, and Dirichlet boundary DOFs.
        self.disc = Discretization(nmesh=nmesh, mesh_type=mesh_type, degree=1)
        self.mesh = self.disc.mesh
        self.V = self.disc.V
        self.bottom_dofs = self.disc.bottom_dofs

        # Define thermal conductivity and heat source as domain coefficients.
        self.h = ThermalConductivity(
            h, self.mesh, self.V, tab_interpolator=tab_interpolator
        )
        self.q = HeatSource(q, self.mesh, self.V, tab_interpolator=tab_interpolator)

        # Define domain boundary conditions:
        ## 1) Dirichlet BC at the bottom.
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(DBC_value), self.bottom_dofs, self.V)
        ]
        ## 2) Neumann BC on the other three edges (insulated, zero flux).
        ## Note: No explicit Neumann BC is needed in the weak form.
```

Delete the now-unused `x = ufl.SpatialCoordinate(self.mesh)` line if it is still present — it was never used.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_smoke.py -v
pytest -q
```

Expected: 5 passed in the smoke module; `27 passed` overall.

- [ ] **Step 5: Confirm numbers are unchanged**

```bash
pytest -q -m gradcheck
```

Expected: `3 passed`. This tier is behavior-preserving — if a gradient check moves, the refactor changed something it should not have.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/forward_solver.py tests/test_smoke.py
git commit -m "Hold a Discretization in the forward solver"
```

---

### Task 3: Compose the adjoint solver instead of inheriting

`SteadyHeat2DAdjointSolver` subclasses `SteadyHeat2DForwardSolver` but never calls `super().__init__()`; it hand-copies six attributes. The subclass therefore inherits methods that are wrong on it — `export_xdmf()` writes the forward temperature and never `lambda_L`, `add_noise()` would overwrite `T_obs` with noise on the forward field, `plot_output_temperature()` plots the forward solution — and `isinstance(adj, SteadyHeat2DForwardSolver)` returns `True`, so any multi-fidelity dispatch keyed on solver type mis-routes.

**Files:**
- Modify: `src/adjoint_solver.py`
- Create: `tests/test_solver_structure.py`

**Interfaces:**
- Consumes: `Discretization` (Task 1), `SteadyHeat2DForwardSolver.disc` (Task 2).
- Produces: `SteadyHeat2DAdjointSolver` as a standalone class holding `.disc` and `.fwd`. Tier 3 modifies its right-hand side but not its construction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_solver_structure.py`:

```python
# tests/test_solver_structure.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# dolfinx imports
from dolfinx import fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from discretization import Discretization


def _forward(nmesh=8):
    fwd = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    return fwd


def test_adjoint_is_not_a_forward_solver():
    """Composition, not inheritance: type-based dispatch must not confuse them."""
    fwd = _forward()
    T_obs = fwd.T.x.array.copy()
    adj = SteadyHeat2DAdjointSolver(fwd, T_obs, sigma=1.0, alpha=0.0)

    assert not isinstance(adj, SteadyHeat2DForwardSolver)
    assert isinstance(adj.disc, Discretization)
    assert adj.disc is fwd.disc  # the SAME discretization, not a copy


def test_adjoint_does_not_expose_forward_only_methods():
    """Methods that are meaningless on the adjoint must not be inherited."""
    fwd = _forward()
    adj = SteadyHeat2DAdjointSolver(fwd, fwd.T.x.array.copy(), sigma=1.0, alpha=0.0)

    for name in ("add_noise", "export_xdmf", "plot_output_temperature"):
        assert not hasattr(adj, name), f"adjoint should not expose {name}()"


def test_adjoint_shares_the_conductivity_object():
    """The gradient depends on mutating fwd.h being visible to the adjoint form."""
    fwd = _forward()
    adj = SteadyHeat2DAdjointSolver(fwd, fwd.T.x.array.copy(), sigma=1.0, alpha=0.0)

    assert adj.h is fwd.h
    assert adj.h.function is fwd.h.function


def test_adjoint_reaches_the_same_gradient_as_before():
    """Composition must not perturb the gradient: this tier changes no numbers."""
    fwd = _forward(nmesh=16)
    T_obs = fwd.T.x.array.copy()
    adj = SteadyHeat2DAdjointSolver(fwd, T_obs, sigma=1.0, alpha=1e-6)
    adj.solve()
    adj.update_gradient()

    # A zero residual leaves only the regularization term in the gradient.
    assert np.isfinite(adj.grad.array).all()
```

Note the deliberate omission: there is no test here for `T_obs` validation. The bare `assert T_obs.function_space == self.V` stays as-is through this tier, because Tier 3 removes `T_obs` altogether in favour of a sensor-space observation vector. Fixing it now would be work Tier 3 discards.

These assertions are all about *composition*, so they survive Tier 3 verbatim — only the constructor call in `_forward`-style helpers changes when the signature does.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_solver_structure.py -v
```

Expected: `test_adjoint_is_not_a_forward_solver`, `test_adjoint_does_not_expose_forward_only_methods`, and both `ValueError` tests FAIL. `test_adjoint_shares_the_conductivity_object` PASSES already — it is a regression guard protecting the shared-handle contract the gradient depends on.

- [ ] **Step 3: Rewrite the class as a composition**

In `src/adjoint_solver.py`, change the class statement and the attribute-copying block. Replace `class SteadyHeat2DAdjointSolver(SteadyHeat2DForwardSolver):` with `class SteadyHeat2DAdjointSolver:` and replace the copying block with:

```python
        # Share the forward solver's discretization and coefficients. `h` in
        # particular MUST be the same object: the gradient depends on TAO's
        # updates to fwd.h.function being visible to this form.
        self.fwd = forward
        self.disc = forward.disc
        self.mesh = self.disc.mesh
        self.V = self.disc.V
        self.bottom_dofs = self.disc.bottom_dofs
        self.h, self.q = forward.h, forward.q
        self.T = forward.T
        self.petsc_opts = forward.petsc_opts

        # Read observed temperature, noise level and regularization amplitude.
        # NOTE: this block is preserved VERBATIM, including its bare `assert`.
        # Tier 3 replaces T_obs entirely with a sensor-space observation vector and
        # rewrites this validation once. Fixing the assert here would be work Tier 3
        # immediately discards.
        self.T_obs = fem.Function(self.V, name="ObservedTemperature")
        if isinstance(T_obs, fem.Function):
            assert T_obs.function_space == self.V
            self.T_obs.x.array[:] = T_obs.x.array[:]
        else:
            self.T_obs.x.array[:] = np.asarray(T_obs, dtype=float)
        self.T_obs.x.scatter_forward()
        self.sigma2 = sigma**2
        self.alpha = alpha
```

Keep the `from forward_solver import SteadyHeat2DForwardSolver` import — it is still used in the type annotation of the `forward` parameter.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_solver_structure.py -v
pytest -q
```

Expected: 5 passed in the structure module; `32 passed` overall.

- [ ] **Step 5: Confirm the gradient is unchanged**

```bash
pytest -q -m gradcheck
```

Expected: `3 passed`. The adjoint form and gradient assembly were untouched, so the numbers must be identical.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/adjoint_solver.py tests/test_solver_structure.py
git commit -m "Compose the adjoint solver instead of inheriting"
```

---

### Task 4: Remove the P1-only DOF-count assumption

`mesh.geometry.x.shape[0]` is used as the local DOF count in five places. It coincides with the P1 DOF count only by construction, and DOF ordering is not guaranteed to match geometry-node ordering. A multi-fidelity hierarchy using P1/P2 pairs would make these slices silently truncate or mis-associate values with no error.

**Files:**
- Modify: `src/forward_solver.py` (`add_noise`, `plot_output_temperature`)
- Modify: `src/domain_coefficient.py` (both plot helpers)

**Interfaces:**
- Consumes: `Discretization.n_dofs_local` (Task 1), `self.disc` (Task 2).
- Produces: no new symbols. Tier 3 replaces `add_noise` entirely but inherits the corrected slicing pattern in the plot helpers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_discretization.py`:

```python
def test_plot_slicing_uses_dof_counts_not_geometry_nodes():
    """A P2 space has more dofs than mesh nodes; slicing by geometry truncates."""
    sys.path.insert(0, "src")
    from discretization import Discretization

    d = Discretization(nmesh=8, mesh_type="quadrilateral", degree=2)

    n_geometry_nodes = d.mesh.geometry.x.shape[0]
    assert d.n_dofs_local != n_geometry_nodes, (
        "test is vacuous unless the two counts differ"
    )
    # Slicing a dof array by the geometry count would silently drop values.
    assert d.n_dofs_local > n_geometry_nodes
```

- [ ] **Step 2: Run it to confirm the premise**

```bash
pytest tests/test_discretization.py -v -k plot_slicing
```

Expected: PASS. This test states the invariant that motivates the change; the change itself is verified by Step 5's equivalence check.

- [ ] **Step 3: Replace the slicing in `forward_solver.py`**

In `add_noise`, replace the DOF-count line:

```python
        # Number of DOFs addressable on this rank (owned + ghosts).
        n_local_dofs = self.disc.n_dofs_local
```

In `plot_output_temperature`, replace both branches' slicing:

```python
        if noiseless:
            if not hasattr(self, "T"):
                raise RuntimeError("No solution available. Call solve() first.")
            vals = self.T.x.array[: self.disc.n_dofs_local] - zero_point
        else:
            if not hasattr(self, "T_obs"):
                raise RuntimeError(
                    "No noisy observation available. Call add_noise() first."
                )
            vals = self.T_obs.x.array[: self.disc.n_dofs_local] - zero_point
```

- [ ] **Step 4: Replace the slicing in `domain_coefficient.py`**

Both `plot_input_thermal_conductivity` and `plot_input_heat_source` use the same pattern. In each, replace:

```python
            vals = self.function.x.array[: self._mesh.geometry.x.shape[0]]
```

with:

```python
            index_map = self._V.dofmap.index_map
            vals = self.function.x.array[: index_map.size_local + index_map.num_ghosts]
```

`BaseDomainCoefficient` holds `self._V` but no `Discretization`, so it reads the index map directly rather than gaining a new dependency.

- [ ] **Step 5: Verify the P1 results are numerically identical**

For P1 the two counts coincide, so every existing number must be unchanged:

```bash
pytest -q
pytest -q -m gradcheck
```

Expected: `33 passed`, then `3 passed`, with no value changes.

- [ ] **Step 6: Verify rendering still works**

```bash
PYVISTA_OFF_SCREEN=true python -c "
import sys; sys.path.insert(0,'src')
from forward_solver import SteadyHeat2DForwardSolver
f = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0+x[0], q=1.0); f.solve()
f.add_noise(0.0, 1e-3, seed=0)
f.plot_output_temperature(zero_point=0.0, noiseless=False)
f.h.plot_input_thermal_conductivity()
print('plots OK')
"
```

Expected: `plots OK`.

- [ ] **Step 7: Commit**

```bash
black src tests
git add src/forward_solver.py src/domain_coefficient.py tests/test_discretization.py
git commit -m "Use index-map dof counts instead of geometry node counts"
```

---

### Task 5: Correct the `h` callable contract and de-duplicate the tangent solver

Two loose ends that would otherwise mislead Tier 3's implementer.

`forward_solver.py`'s docstring promises `callable(x,y)` interpolated via `fem.Function.interpolate`, but `domain_coefficient.py` actually calls the user's function on a `ufl.SpatialCoordinate`. A NumPy-style callable such as `lambda x: np.sin(x[0])` — the form the docstring describes — raises inside UFL. (`docs/usage.md` already documents the real behavior; only the source docstring is wrong.)

`tests/_tangent_solver.py` duplicates the adjoint solver's old attribute-copying block verbatim, which Task 3 has now removed from the adjoint.

**Files:**
- Modify: `src/forward_solver.py` (docstring only)
- Modify: `tests/_tangent_solver.py`

**Interfaces:**
- Consumes: `SteadyHeat2DForwardSolver.disc` (Task 2).
- Produces: no new symbols.

- [ ] **Step 1: Correct the docstring**

In `src/forward_solver.py`, replace the `h` and `q` entries of the `__init__` Parameters block:

```python
        nmesh      : number of cells per side of the unit square, i.e. (x,y) in [0,1].
        h          : thermal conductivity, accepting any of
                     - float or fem.Constant, giving a fem.Constant on the domain;
                     - a callable of the UFL spatial coordinate, e.g.
                       ``lambda x: 1.0 + 6.0 * x[0] ** 2``, evaluated symbolically and
                       interpolated onto the domain. It must be UFL-expressible: a
                       NumPy-style callable such as ``lambda x: np.sin(x[0])`` will not
                       work, because the argument is a ufl.SpatialCoordinate;
                     - a fem.Expression, interpolated directly;
                     - a tabulated (N,3) array or (x|y|value) DataFrame, interpolated
                       per `tab_interpolator`.
        q          : heat source, same allowed types as h.
```

- [ ] **Step 2: Verify the docstring renders**

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Expected: `build succeeded.` with zero warnings. The list under a `:`-terminated lead-in must keep its existing indentation, or reST will reject it.

- [ ] **Step 3: Share the discretization in the tangent solver**

In `tests/_tangent_solver.py`, replace the hand-copied attribute block with the shared objects:

```python
        # Share the forward solver's discretization and coefficients.
        self.fwd = forward
        self.disc = forward.disc
        self.mesh = self.disc.mesh
        self.V = self.disc.V
        self.bottom_dofs = self.disc.bottom_dofs
        self.h, self.q = forward.h, forward.q
        self.T = forward.T
        self.petsc_opts = forward.petsc_opts
```

If the class currently subclasses `SteadyHeat2DForwardSolver`, change it to a standalone `class` statement as in Task 3. Leave its weak form and solve method untouched.

- [ ] **Step 4: Verify the forward-mode gradient check still holds**

```bash
pytest -q -m gradcheck
```

Expected: `3 passed`. `test_grad_forwarddiff.py` is the direct consumer of the tangent solver; an identical result confirms the de-duplication changed nothing.

- [ ] **Step 5: Run the full suite**

```bash
pytest -q
```

Expected: `33 passed`.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/forward_solver.py tests/_tangent_solver.py
git commit -m "Correct the h callable contract and share the discretization in the tangent solver"
```

---

## Tier 2 Verification Round

This tier's defining property is that it changed no numbers. Verify that explicitly, not just that tests pass.

- [ ] **Full suite passes**

```bash
pytest -q
```
Expected: `33 passed`.

- [ ] **Gradient values are numerically unchanged**

Compare against Tier 1's values by re-running the directional-derivative check and printing the number rather than only asserting on it:

```bash
python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.')
import numpy as np
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tests._helpers import pick_random_test_direction, h_true, h0

t = SteadyHeat2DForwardSolver(nmesh=16, h=h_true, q=1.0, DBC_value=300.0); t.solve()
f = SteadyHeat2DForwardSolver(nmesh=16, h=h0, q=1.0, DBC_value=300.0); f.solve()
a = SteadyHeat2DAdjointSolver(f, t.T.x.array.copy(), sigma=1.0, alpha=1e-6, DBC_value=0.0)
a.solve(); a.update_gradient()
d = pick_random_test_direction(f.V, seed=244, scale=1.0)
print('directional derivative: %.15e' % a.grad.dot(d.x.petsc_vec))
"
```

Record the printed value. Re-run this exact command after every task in this tier — it must not change in any digit. If it does, the refactor altered behavior and the task must be revisited before proceeding.

- [ ] **Composition contract holds**

```bash
python -c "
import sys; sys.path.insert(0,'src')
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
f = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0+x[0], q=1.0); f.solve()
a = SteadyHeat2DAdjointSolver(f, f.T.x.array.copy(), sigma=1.0, alpha=0.0)
print('adjoint is a forward solver:', isinstance(a, SteadyHeat2DForwardSolver))
print('shares discretization:', a.disc is f.disc)
print('shares h object:', a.h is f.h)
"
```
Expected: `False`, `True`, `True`.

- [ ] **P2 spaces construct correctly**

```bash
python -c "
import sys; sys.path.insert(0,'src')
from discretization import Discretization
d = Discretization(nmesh=8, degree=2)
print('P2 dofs:', d.n_dofs_local, 'geometry nodes:', d.mesh.geometry.x.shape[0])
assert d.n_dofs_local > d.mesh.geometry.x.shape[0]
print('index-map counts correct')
"
```
Expected: the DOF count exceeds the node count.

- [ ] **Docs build clean**

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: `build succeeded.`

- [ ] **CI green**

```bash
git push origin master && sleep 90 && gh run list --limit 4
```
Expected: no `failure`.

**Exit criteria:** every box checked, and the directional derivative identical to Tier 1's value. Tier 3 assumes `Discretization`, a composed adjoint solver, and index-map DOF counts.
