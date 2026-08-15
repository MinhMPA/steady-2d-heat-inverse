# Tier 3: Inference-Ready Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the objective an actual log-posterior of an explicit, mesh-independent observation model, and expose it as a pure `value_and_grad` boundary that both PETSc TAO and BlackJAX/NUTS can consume — unblocking multi-fidelity optimization and HMC.

**Architecture:** A `SensorOperator` `B` interpolates the temperature field to fixed physical measurement points, decoupling the data from the discretization. The likelihood becomes `½|B·T − d|²/σ²` with i.i.d. sensor noise, so coarse and fine meshes predict the *same* data vector. A new `objective` module exposes `value_and_grad(m)` — MPI-reduced, form-cached, and carrying the log-`h` change-of-variables Jacobian — with TAO's `_objgrad` reduced to a thin adapter.

**Tech Stack:** DOLFINx 0.11, PETSc/TAO, SciPy sparse, pytest. (BlackJAX/JAX enter in a follow-on plan; this tier delivers the sampler-neutral boundary they need.)

## Prerequisites

**Tiers 1 and 2 must be complete with their verification rounds green.** This tier assumes: pinned DOLFINx 0.11, green CI, lazy `pyvista`, a `Discretization` value object, a *composed* (non-inheriting) adjoint solver, and index-map DOF counts.

## The decision this tier implements

Recorded when the plan was commissioned: **`Σ = σ²I` over fixed sensors**, i.e. an explicit observation operator rather than treating every DOF as a measurement.

Consequences, stated plainly so nobody re-litigates them mid-implementation:

- **α must be re-tuned.** `J`'s scale changes; every previously recorded α is meaningless. This is why Tier 1 deleted the sweep artifacts rather than regenerating them.
- **The adjoint RHS changes character** — from a smooth integrated source `(T − T_obs)/σ²` to a sum of point loads `Bᵀr/σ²`. This forces the adjoint solver off `LinearProblem` and onto explicit matrix assembly plus a KSP, because the RHS is now a vector rather than a UFL form.
- **`add_noise` is replaced.** Observations live in sensor space, not DOF space.

## Global Constraints

- **Import convention:** flat top-level modules. Always `from objective import ...`, never `from src.objective import ...`.
- **Test imports:** `tests/` is a package. Import shared helpers relatively; import code under test flatly.
- **Formatting:** `black`. Run `black src tests scripts` before each commit.
- **Import banners:** group imports under the existing comment banners (`# type imports`, `# numerical imports`, `# mpi imports`, `# pde imports`, `# dolfinx imports`, `# local imports`).
- **Docstrings:** raw strings (`r"""`) with LaTeX-style math; document Parameters and Returns.
- **Docstring reST rule:** an indented equation under a `:`-terminated lead-in must be followed by a blank line before the next unindented paragraph, or `sphinx-build -W` fails.
- **MPI hygiene:** after mutating `.x.array`, call `scatter_forward()`; after assembling a vector, `ghostUpdate()`. Every scalar that represents a global integral or sum **must** be `allreduce`d — that is the specific bug this tier exists to prevent recurring.
- **Commits:** short imperative subject lines, no `feat:`/`fix:` prefixes.
- **Register every new module** in `pyproject.toml` `py-modules` and re-run `pip install -e .`, or the flat import will fail.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/sensors.py` | create | `SensorOperator` — assemble `B` and `Bᵀ` |
| `src/observation.py` | create | `GaussianSensorNoise` — sample `d`, weight residuals, log-likelihood |
| `src/objective.py` | create | `Objective.value_and_grad(m)` — pure, MPI-reduced, log-`h` Jacobian |
| `src/adjoint_solver.py` | modify | vector RHS via explicit assembly + KSP |
| `src/tao_solver.py` | modify | `_objgrad` becomes a thin adapter over `Objective` |
| `src/forward_solver.py` | modify | remove `add_noise` (superseded by the observation model) |
| `scripts/make_fixture.py` | modify | emit sensor observations `d` alongside the field |
| `tests/test_sensors.py` | create | `B` exactness, partition of unity, adjoint identity |
| `tests/test_observation.py` | create | sampling determinism, log-likelihood |
| `tests/test_objective.py` | create | finite-difference gradient, Jacobian, MPI reduction |
| `tests/test_mesh_independence.py` | create | the payoff test: same data across fidelities |

---

### Task 1: The sensor operator

`B` maps FEM degrees of freedom to sensor readings: `(B @ T)[k] = T(x_k, y_k)`. Assembled once as a SciPy sparse matrix so that `B.T` — needed for every adjoint solve — is free.

The assembly below is **verified working code**: on a 16×16 quadrilateral P1 mesh it reproduces a linear field to 2.7e-15, its rows sum to 1 within 2.2e-16, and the adjoint identity `⟨Bf, r⟩ = ⟨f, Bᵀr⟩` closes to 8.9e-16.

**Files:**
- Create: `src/sensors.py`
- Create: `tests/test_sensors.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `Discretization` (Tier 2 Task 1).
- Produces: `SensorOperator(disc, points)` with `.matrix` (CSR, shape `(n_sensors, n_dofs_local)`), `.n_sensors`, `.points`, `.apply(dof_array) -> sensor_array`, `.apply_transpose(sensor_array) -> dof_array`. Tasks 2–5 use it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sensors.py`:

```python
# tests/test_sensors.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# dolfinx imports
from dolfinx import fem

# local imports
from discretization import Discretization
from sensors import SensorOperator


def _grid_points(n_side=5):
    """A regular sensor grid strictly inside the unit square."""
    g = np.linspace(0.1, 0.9, n_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_interpolates_a_linear_field_exactly():
    """P1 reproduces linear fields, so B must be exact to round-off."""
    disc = Discretization(nmesh=16, mesh_type="quadrilateral")
    f = fem.Function(disc.V)
    f.interpolate(lambda x: 1.0 + 2.0 * x[0] + 3.0 * x[1])
    f.x.scatter_forward()

    pts = _grid_points()
    B = SensorOperator(disc, pts)

    got = B.apply(f.x.array)
    want = 1.0 + 2.0 * pts[:, 0] + 3.0 * pts[:, 1]

    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_rows_are_a_partition_of_unity():
    """Interpolation weights must sum to 1, or constants are not reproduced."""
    disc = Discretization(nmesh=16, mesh_type="quadrilateral")
    B = SensorOperator(disc, _grid_points())

    row_sums = np.asarray(B.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=0.0, atol=1e-12)


def test_adjoint_identity():
    """<B f, r> == <f, B^T r>: the property the adjoint RHS depends on."""
    disc = Discretization(nmesh=16, mesh_type="quadrilateral")
    B = SensorOperator(disc, _grid_points())

    rng = np.random.default_rng(0)
    f = rng.normal(size=B.matrix.shape[1])
    r = rng.normal(size=B.n_sensors)

    lhs = float(B.apply(f) @ r)
    rhs = float(f @ B.apply_transpose(r))

    assert lhs == pytest.approx(rhs, rel=0.0, abs=1e-10)


def test_shape_and_sparsity():
    """Quadrilateral P1: each sensor lies in one cell with four nodes."""
    disc = Discretization(nmesh=16, mesh_type="quadrilateral")
    pts = _grid_points()
    B = SensorOperator(disc, pts)

    assert B.matrix.shape == (pts.shape[0], disc.n_dofs_local)
    assert B.n_sensors == pts.shape[0]
    assert B.matrix.nnz == 4 * pts.shape[0]


def test_point_outside_the_mesh_raises():
    disc = Discretization(nmesh=8, mesh_type="quadrilateral")
    outside = np.array([[1.5, 0.5]])

    with pytest.raises(ValueError, match="outside the mesh"):
        SensorOperator(disc, outside)


def test_triangular_mesh_is_supported():
    """Triangles give three nodes per cell rather than four."""
    disc = Discretization(nmesh=16, mesh_type="triangle")
    f = fem.Function(disc.V)
    f.interpolate(lambda x: 1.0 + 2.0 * x[0] + 3.0 * x[1])
    f.x.scatter_forward()

    pts = _grid_points()
    B = SensorOperator(disc, pts)

    want = 1.0 + 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
    np.testing.assert_allclose(B.apply(f.x.array), want, rtol=0.0, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_sensors.py -v
```

Expected: all six FAIL with `ModuleNotFoundError: No module named 'sensors'`.

- [ ] **Step 3: Write the module**

Create `src/sensors.py`:

```python
# type imports
from typing import Union

# numerical imports
import numpy as np
import scipy.sparse as sp

# dolfinx imports
from dolfinx import geometry

# local imports
from discretization import Discretization


class SensorOperator:
    r"""
    The observation operator :math:`B` mapping FEM degrees of freedom to sensor readings:

        (B T)_k = T(x_k, y_k)

    Sensor locations are physical and fixed, independent of the mesh. That is what
    makes the likelihood mesh-independent: refining the discretization improves the
    prediction of the same measurements rather than inventing new ones.

    :math:`B` is stored as a CSR matrix, so :math:`B^\mathsf{T}` -- needed for every
    adjoint right-hand side -- costs nothing to form.

    Parameters
    ----------
    disc   : the Discretization whose function space the operator acts on.
    points : (n_sensors, 2) array of physical sensor coordinates.
    """

    def __init__(self, disc: Discretization, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (n_sensors, 2), got {points.shape}."
            )

        self.disc = disc
        self.points = points
        self.n_sensors = points.shape[0]
        self.matrix = self._assemble(disc, points)

    @staticmethod
    def _assemble(disc: Discretization, points: np.ndarray) -> sp.csr_matrix:
        """Build the sparse interpolation matrix by tabulating basis functions."""
        domain = disc.mesh
        V = disc.V
        n_sensors = points.shape[0]

        # DOLFINx geometry queries work in 3D coordinates.
        pts3 = np.zeros((n_sensors, 3), dtype=np.float64)
        pts3[:, :2] = points

        tree = geometry.bb_tree(domain, domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, pts3)
        colliding = geometry.compute_colliding_cells(domain, candidates, pts3)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        element = V.element
        dofmap = V.dofmap

        for k in range(n_sensors):
            cells = colliding.links(k)
            if len(cells) == 0:
                raise ValueError(
                    f"sensor {k} at {points[k]} is outside the mesh "
                    "(or outside this rank's partition)."
                )
            cell = cells[0]

            # Pull the physical point back to the reference cell.
            geom_dofs = domain.geometry.dofmap[cell]
            cell_coords = domain.geometry.x[geom_dofs]
            reference = domain.geometry.cmap.pull_back(
                pts3[k].reshape(1, 3), cell_coords
            )
            # Tabulate the basis (derivative order 0) at that reference point.
            basis = element.basix_element.tabulate(0, reference)[0, 0, :, 0]

            for local_index, global_dof in enumerate(dofmap.cell_dofs(cell)):
                rows.append(k)
                cols.append(global_dof)
                vals.append(float(basis[local_index]))

        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_sensors, disc.n_dofs_local)
        )

    def apply(self, dof_values: np.ndarray) -> np.ndarray:
        """Interpolate a DOF array to sensor readings. Returns (n_sensors,)."""
        return self.matrix @ np.asarray(dof_values, dtype=np.float64)

    def apply_transpose(self, sensor_values: np.ndarray) -> np.ndarray:
        r"""Scatter sensor-space values back to DOF space via :math:`B^\mathsf{T}`.

        This is the adjoint right-hand side: a sum of point loads located at the
        sensors, weighted by the residual.

        Returns
        -------
        (n_dofs_local,) array.
        """
        return self.matrix.T @ np.asarray(sensor_values, dtype=np.float64)
```

- [ ] **Step 4: Register the module**

In `pyproject.toml`, add `"sensors"` to `py-modules`, then:

```bash
pip install -e . -q
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_sensors.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/sensors.py tests/test_sensors.py pyproject.toml
git commit -m "Add the sensor observation operator"
```

---

### Task 2: The Gaussian sensor noise model

With `B` in place, the observation model is `d = B·T_true + ε`, `ε ~ N(0, σ²I)` over sensors. This object owns both directions: drawing synthetic data, and weighting residuals in the likelihood. Keeping them in one class is what prevents the sampling/objective mismatch that motivated this tier.

**Files:**
- Create: `src/observation.py`
- Create: `tests/test_observation.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `SensorOperator` (Task 1).
- Produces: `GaussianSensorNoise(sensors, sigma)` with `.sample(dof_values, seed) -> d`, `.residual(dof_values, d) -> r`, `.neg_log_likelihood(dof_values, d) -> float`, `.adjoint_source(dof_values, d) -> dof_array`. Tasks 3–5 use it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_observation.py`:

```python
# tests/test_observation.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# dolfinx imports
from dolfinx import fem

# local imports
from discretization import Discretization
from sensors import SensorOperator
from observation import GaussianSensorNoise


def _setup(n_side=5, nmesh=16):
    disc = Discretization(nmesh=nmesh, mesh_type="quadrilateral")
    g = np.linspace(0.1, 0.9, n_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    B = SensorOperator(disc, pts)

    f = fem.Function(disc.V)
    f.interpolate(lambda x: 300.0 + x[0] + x[1])
    f.x.scatter_forward()
    return disc, B, f


def test_sampling_is_reproducible_from_a_seed():
    _, B, f = _setup()
    obs = GaussianSensorNoise(B, sigma=1e-2)

    d1 = obs.sample(f.x.array, seed=0)
    d2 = obs.sample(f.x.array, seed=0)
    d3 = obs.sample(f.x.array, seed=1)

    np.testing.assert_array_equal(d1, d2)
    assert not np.array_equal(d1, d3)


def test_sample_shape_is_sensor_space_not_dof_space():
    disc, B, f = _setup()
    obs = GaussianSensorNoise(B, sigma=1e-2)

    d = obs.sample(f.x.array, seed=0)

    assert d.shape == (B.n_sensors,)
    assert d.size != disc.n_dofs_local  # the distinction this tier exists for


def test_noise_has_the_requested_scale():
    _, B, f = _setup(n_side=30)
    sigma = 0.25
    obs = GaussianSensorNoise(B, sigma=sigma)

    d = obs.sample(f.x.array, seed=0)
    noise = d - B.apply(f.x.array)

    assert noise.std() == pytest.approx(sigma, rel=0.15)


def test_zero_residual_gives_zero_negative_log_likelihood():
    _, B, f = _setup()
    obs = GaussianSensorNoise(B, sigma=1e-2)
    d = B.apply(f.x.array)  # noiseless

    assert obs.neg_log_likelihood(f.x.array, d) == pytest.approx(0.0, abs=1e-20)


def test_negative_log_likelihood_matches_the_closed_form():
    _, B, f = _setup()
    sigma = 0.1
    obs = GaussianSensorNoise(B, sigma=sigma)
    d = obs.sample(f.x.array, seed=0)

    r = B.apply(f.x.array) - d
    expected = 0.5 * float(r @ r) / sigma**2

    assert obs.neg_log_likelihood(f.x.array, d) == pytest.approx(expected, rel=1e-12)


def test_adjoint_source_is_b_transpose_of_the_weighted_residual():
    _, B, f = _setup()
    sigma = 0.1
    obs = GaussianSensorNoise(B, sigma=sigma)
    d = obs.sample(f.x.array, seed=0)

    got = obs.adjoint_source(f.x.array, d)
    want = B.apply_transpose((B.apply(f.x.array) - d) / sigma**2)

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_observation.py -v
```

Expected: all six FAIL with `ModuleNotFoundError: No module named 'observation'`.

- [ ] **Step 3: Write the module**

Create `src/observation.py`:

```python
# numerical imports
import numpy as np

# local imports
from sensors import SensorOperator


class GaussianSensorNoise:
    r"""
    Independent Gaussian measurement error at fixed sensors:

        d = B T + \varepsilon,   \varepsilon \sim N(0, \sigma^2 I)

    The negative log-likelihood is therefore

        -\log p(d \mid T) = \frac{1}{2\sigma^2} \lVert B T - d \rVert^2

    up to an additive constant independent of :math:`T`.

    Sampling and weighting live in one object deliberately: when they are defined
    separately they drift apart, and the objective silently stops being the
    likelihood of the data it was given.

    Parameters
    ----------
    sensors : the SensorOperator defining B.
    sigma   : standard deviation of the measurement error, per sensor.
    """

    def __init__(self, sensors: SensorOperator, sigma: float):
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        self.sensors = sensors
        self.sigma = float(sigma)
        self.sigma2 = float(sigma) ** 2

    def sample(self, dof_values: np.ndarray, seed: int | None = None) -> np.ndarray:
        r"""
        Draw synthetic observations :math:`d = B T + \varepsilon`.

        Parameters
        ----------
        dof_values : the (noiseless) temperature DOF array.
        seed       : RNG seed. Pass one: an unseeded draw makes the inverse
                     problem's target irreproducible.

        Returns
        -------
        (n_sensors,) array of observations.
        """
        rng = np.random.default_rng(seed)
        clean = self.sensors.apply(dof_values)
        return clean + rng.normal(loc=0.0, scale=self.sigma, size=clean.size)

    def residual(self, dof_values: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Sensor-space residual ``B T - d``. Returns (n_sensors,)."""
        return self.sensors.apply(dof_values) - np.asarray(d, dtype=np.float64)

    def neg_log_likelihood(self, dof_values: np.ndarray, d: np.ndarray) -> float:
        r"""Return :math:`\tfrac{1}{2\sigma^2}\lVert BT - d\rVert^2`."""
        r = self.residual(dof_values, d)
        return 0.5 * float(r @ r) / self.sigma2

    def adjoint_source(self, dof_values: np.ndarray, d: np.ndarray) -> np.ndarray:
        r"""
        The derivative :math:`\partial J / \partial T` as a DOF-space vector.

        This is :math:`B^\mathsf{T}(BT - d)/\sigma^2` -- a sum of point loads at the
        sensor locations, which becomes the adjoint equation's right-hand side.

        Returns
        -------
        (n_dofs_local,) array.
        """
        r = self.residual(dof_values, d)
        return self.sensors.apply_transpose(r / self.sigma2)
```

- [ ] **Step 4: Register and install**

Add `"observation"` to `py-modules` in `pyproject.toml`, then `pip install -e . -q`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest tests/test_observation.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/observation.py tests/test_observation.py pyproject.toml
git commit -m "Add the Gaussian sensor noise model"
```

---

### Task 3: Drive the adjoint from a vector right-hand side

The adjoint source is now a DOF-space vector (`Bᵀr/σ²`), not a UFL form. `LinearProblem` takes a form for its RHS, so the adjoint solver must assemble the matrix once and solve with an explicitly-constructed vector.

**Files:**
- Modify: `src/adjoint_solver.py`
- Create: `tests/test_adjoint_vector_rhs.py`

**Interfaces:**
- Consumes: `GaussianSensorNoise` (Task 2), the composed adjoint solver (Tier 2 Task 3).
- Produces: `SteadyHeat2DAdjointSolver(forward, observation, d, alpha=...)` with `.solve()`, `.assemble_gradient()`, `.update_gradient()`, `.grad`. Tasks 4–5 use it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adjoint_vector_rhs.py`:

```python
# tests/test_adjoint_vector_rhs.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from sensors import SensorOperator
from observation import GaussianSensorNoise


def _sensor_grid(n_side=5):
    g = np.linspace(0.1, 0.9, n_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _setup(nmesh=16, sigma=1e-2, alpha=0.0):
    truth = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    truth.solve()
    B = SensorOperator(truth.disc, _sensor_grid())
    obs = GaussianSensorNoise(B, sigma=sigma)
    d = obs.sample(truth.T.x.array, seed=0)

    fwd = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    fwd_B = SensorOperator(fwd.disc, _sensor_grid())
    fwd_obs = GaussianSensorNoise(fwd_B, sigma=sigma)
    adj = SteadyHeat2DAdjointSolver(fwd, fwd_obs, d, alpha=alpha)
    return fwd, adj, d


def test_adjoint_solves_and_respects_the_dirichlet_condition():
    """lambda(y=0) = 0 is the adjoint boundary condition."""
    fwd, adj, _ = _setup()
    lam = adj.solve()

    assert np.isfinite(lam.x.array).all()
    np.testing.assert_allclose(
        lam.x.array[fwd.disc.bottom_dofs], 0.0, rtol=0.0, atol=1e-10
    )


def test_zero_residual_gives_zero_adjoint_state():
    """With d taken noiselessly from this same forward state, the source vanishes."""
    fwd = SteadyHeat2DForwardSolver(
        nmesh=16, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    B = SensorOperator(fwd.disc, _sensor_grid())
    obs = GaussianSensorNoise(B, sigma=1e-2)
    d_exact = B.apply(fwd.T.x.array)

    adj = SteadyHeat2DAdjointSolver(fwd, obs, d_exact, alpha=0.0)
    lam = adj.solve()

    np.testing.assert_allclose(lam.x.array, 0.0, rtol=0.0, atol=1e-9)


def test_gradient_is_assembled_and_finite():
    _, adj, _ = _setup(alpha=1e-6)
    adj.solve()
    adj.update_gradient()

    g = adj.grad
    assert np.isfinite(g.array).all()


def test_grad_before_update_raises():
    _, adj, _ = _setup()
    with pytest.raises(RuntimeError, match="update_gradient"):
        _ = adj.grad
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_adjoint_vector_rhs.py -v
```

Expected: all four FAIL — the constructor signature does not yet accept `(forward, observation, d, alpha=...)`.

- [ ] **Step 3: Rewrite the adjoint solver**

Replace the contents of `src/adjoint_solver.py`:

```python
# numerical imports
import numpy as np

# pde imports
from petsc4py import PETSc
import ufl

# dolfinx imports
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from observation import GaussianSensorNoise


class SteadyHeat2DAdjointSolver:
    r"""
    Adjoint solver for the steady-state Poisson heat equation on the unit square.

    The adjoint state :math:`\lambda` satisfies

        -\nabla\cdot(h\nabla\lambda) = \partial J/\partial T,

    with :math:`\lambda(y=0)=0` and zero-flux Neumann conditions elsewhere. The
    operator is self-adjoint, so the left-hand side is identical to the forward one.

    The right-hand side is a DOF-space *vector* rather than a UFL form, because the
    observation operator produces point loads at the sensor locations:

        \partial J/\partial T = B^\mathsf{T}(BT - d)/\sigma^2

    Parameters
    ----------
    forward     : the forward solver, supplying the discretization, h, q and T.
    observation : the GaussianSensorNoise model defining B and sigma.
    d           : (n_sensors,) observed data.
    alpha       : Tikhonov (H^1) regularization weight on h.
    petsc_opts  : PETSc KSP options; inherits the forward solver's when None.
    """

    def __init__(
        self,
        forward: SteadyHeat2DForwardSolver,
        observation: GaussianSensorNoise,
        d: np.ndarray,
        alpha: float = 0.0,
        petsc_opts: dict | None = None,
    ):
        # Share the forward solver's discretization and coefficients. `h` must be
        # the SAME object so TAO's updates are visible to this form.
        self.fwd = forward
        self.disc = forward.disc
        self.mesh = self.disc.mesh
        self.V = self.disc.V
        self.bottom_dofs = self.disc.bottom_dofs
        self.h, self.q = forward.h, forward.q
        self.T = forward.T

        self.obs = observation
        d = np.asarray(d, dtype=np.float64)
        if d.shape != (observation.sensors.n_sensors,):
            raise ValueError(
                f"d has shape {d.shape}, expected "
                f"({observation.sensors.n_sensors},) -- one value per sensor."
            )
        self.d = d
        self.alpha = alpha
        self.petsc_opts = petsc_opts if petsc_opts is not None else forward.petsc_opts

        # Dirichlet condition: lambda(y=0) = 0.
        self.bcs = [
            fem.dirichletbc(PETSc.ScalarType(0.0), self.bottom_dofs, self.V)
        ]

        # Left-hand side: identical to the forward operator (self-adjointness).
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        self.a = ufl.inner(self.h.function * ufl.grad(u), ufl.grad(v)) * ufl.dx
        self._a_form = fem.form(self.a)

        self.lambda_L = fem.Function(self.V, name="AdjointState")

        # KSP configured once; the matrix is re-assembled each solve because h changes.
        self._A = assemble_matrix(self._a_form, bcs=self.bcs)
        self._A.assemble()
        self._b = self.lambda_L.x.petsc_vec.duplicate()

        self._ksp = PETSc.KSP().create(self.mesh.comm)
        self._ksp.setOperators(self._A)
        opts = PETSc.Options()
        prefix = "s2dhi_adjvec_"
        self._ksp.setOptionsPrefix(prefix)
        for key, value in self.petsc_opts.items():
            opts[prefix + key] = value
        self._ksp.setFromOptions()

    def solve(self) -> fem.Function:
        r"""
        Solve the adjoint equation for the current :math:`h` and :math:`T`.

        Returns
        -------
        lambda_L : the adjoint state.
        """
        # h changed since construction, so re-assemble the operator.
        self._A.zeroEntries()
        assemble_matrix(self._A, self._a_form, bcs=self.bcs)
        self._A.assemble()

        # Right-hand side: B^T (B T - d) / sigma^2, with the Dirichlet rows zeroed.
        source = self.obs.adjoint_source(self.T.x.array, self.d)
        with self._b.localForm() as b_local:
            b_local.array[:] = source
        self._b.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        set_bc(self._b, self.bcs)

        self._ksp.solve(self._b, self.lambda_L.x.petsc_vec)
        reason = self._ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"adjoint KSP failed to converge (reason {reason}); "
                "see https://petsc.org/release/manualpages/KSP/KSPConvergedReason/"
            )
        self.lambda_L.x.scatter_forward()
        return self.lambda_L

    def assemble_gradient(self) -> PETSc.Vec:
        r"""
        Assemble :math:`\mathrm{d}J/\mathrm{d}h`.

        The total derivative is

            dJ/dh[v] = \int_\Omega (-\nabla T\cdot\nabla\lambda\, v
                       + \alpha\,\nabla h\cdot\nabla v)\ \mathrm{d}x

        Returns
        -------
        PETSc.Vec : the assembled gradient, ghost-reduced.
        """
        v = ufl.TestFunction(self.V)
        grad_expr = (
            -ufl.inner(ufl.grad(self.T), ufl.grad(self.lambda_L)) * v
            + self.alpha * ufl.inner(ufl.grad(self.h.function), ufl.grad(v))
        ) * ufl.dx
        grad_vec = assemble_vector(fem.form(grad_expr))
        grad_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return grad_vec

    def update_gradient(self) -> None:
        """Recompute and cache the gradient in ``self._grad``."""
        self._grad = self.assemble_gradient()

    @property
    def grad(self) -> PETSc.Vec:
        if not hasattr(self, "_grad"):
            raise RuntimeError("Call update_gradient() first.")
        return self._grad
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_adjoint_vector_rhs.py -v
```

Expected: 4 passed.

If `test_zero_residual_gives_zero_adjoint_state` fails, the RHS assembly or the boundary handling is wrong — that test is the sharpest check on both, because a zero source must give exactly a zero state.

- [ ] **Step 5: Update the existing gradient tests to the new constructor**

`tests/test_grad_finitediff.py`, `tests/test_grad_forwarddiff.py`, and `tests/test_grad_taylorexp.py` construct the adjoint with `(fwd, T_obs, sigma=..., alpha=..., DBC_value=...)`. They now need a sensor operator and observation model. In each file, replace the adjoint construction with:

```python
    # Sensor observations: a fixed grid, independent of the mesh.
    g = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    sensor_pts = np.column_stack([xx.ravel(), yy.ravel()])
    B = SensorOperator(fwd.disc, sensor_pts)
    obs = GaussianSensorNoise(B, sigma=noise_sigma)
    d = B.apply(fwd_truth.T.x.array)  # noiseless observations for the gradient check

    adj = SteadyHeat2DAdjointSolver(fwd, obs, d, alpha=reg_alpha)
```

and add to each file's `# local imports` banner:

```python
from sensors import SensorOperator
from observation import GaussianSensorNoise
```

`tests/test_solver_structure.py` (from Tier 2 Task 3) also constructs the adjoint with the old `(fwd, T_obs, sigma=..., alpha=...)` signature. Update its `SteadyHeat2DAdjointSolver(...)` calls the same way. This is a planned signature update, not a redo: Tier 2 deliberately confined that file to assertions about *composition* (shared discretization, shared `h`, absent forward-only methods), all of which survive verbatim — only the construction line changes.

Also update `tests/_helpers.py`'s `eval_obj` to the new objective — it is replaced wholesale in Task 4, so for now delete it and have the three tests import `evaluate_objective` from Task 4 instead. **Do Task 4 before re-running the gradient checks**; they are verified at the end of Task 4, not here.

DOLFINx 0.11 API note: this task uses `assemble_matrix(form, bcs=...)` to create the operator and `assemble_matrix(A, form, bcs=...)` to re-assemble into it. If the two-argument in-place form is unavailable in the installed version, substitute `A.zeroEntries()` followed by a fresh `assemble_matrix(form, bcs=...)` and `setOperators` — and record which you used in your report.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/adjoint_solver.py tests/test_adjoint_vector_rhs.py
git commit -m "Drive the adjoint from a vector right-hand side"
```

---

### Task 4: The pure, MPI-reduced objective

This is the boundary both TAO and BlackJAX consume. It takes and returns plain arrays, reduces every global scalar across ranks, caches its forms, and optionally carries the log-`h` change-of-variables Jacobian.

Three defects are fixed here by construction rather than by patching: the missing `allreduce` on `J`, the per-evaluation `fem.form` rebuild, and the absent Jacobian term.

**Files:**
- Create: `src/objective.py`
- Create: `tests/test_objective.py`
- Modify: `pyproject.toml`
- Modify: `tests/_helpers.py` (remove the superseded `eval_obj`)

**Interfaces:**
- Consumes: the forward solver, the rewritten adjoint solver (Task 3), `GaussianSensorNoise` (Task 2).
- Produces: `Objective(forward, adjoint, use_logh=True)` with `.value(m) -> float`, `.value_and_grad(m) -> (float, np.ndarray)`, `.n_parameters`. Task 5 adapts TAO onto it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_objective.py`:

```python
# tests/test_objective.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from sensors import SensorOperator
from observation import GaussianSensorNoise
from objective import Objective


def _sensor_grid(n_side=4):
    g = np.linspace(0.15, 0.85, n_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _objective(nmesh=8, alpha=1e-6, sigma=1e-2, use_logh=True):
    truth = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    truth.solve()
    B_t = SensorOperator(truth.disc, _sensor_grid())
    d = B_t.apply(truth.T.x.array)

    fwd = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    B = SensorOperator(fwd.disc, _sensor_grid())
    obs = GaussianSensorNoise(B, sigma=sigma)
    adj = SteadyHeat2DAdjointSolver(fwd, obs, d, alpha=alpha)
    return Objective(fwd, adj, use_logh=use_logh), fwd


def test_value_and_grad_shapes():
    obj, fwd = _objective()
    m = np.log(fwd.h.function.x.array.copy())

    J, g = obj.value_and_grad(m)

    assert isinstance(J, float)
    assert g.shape == m.shape
    assert np.isfinite(J)
    assert np.isfinite(g).all()


def test_gradient_matches_central_finite_differences():
    """The core correctness check for the new objective."""
    obj, fwd = _objective(alpha=1e-6)
    m0 = np.log(fwd.h.function.x.array.copy())

    rng = np.random.default_rng(7)
    direction = rng.normal(size=m0.size)
    direction /= np.linalg.norm(direction)

    _, g = obj.value_and_grad(m0)
    analytic = float(g @ direction)

    eps = 1e-6
    J_plus = obj.value(m0 + eps * direction)
    J_minus = obj.value(m0 - eps * direction)
    numeric = (J_plus - J_minus) / (2.0 * eps)

    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_value_is_deterministic():
    obj, fwd = _objective()
    m = np.log(fwd.h.function.x.array.copy())

    assert obj.value(m) == obj.value(m)


def test_logh_jacobian_is_included_when_requested():
    r"""Sampling in m = log h needs the +sum(m) change-of-variables term.

    Optimization deliberately omits it; sampling must not.
    """
    obj, fwd = _objective(use_logh=True)
    m = np.log(fwd.h.function.x.array.copy())

    J_map = obj.value(m)
    J_post = obj.value(m, include_jacobian=True)

    assert J_post == pytest.approx(J_map - float(np.sum(m)), rel=1e-12)


def test_jacobian_gradient_is_consistent():
    """d/dm of -sum(m) is -1 in every component."""
    obj, fwd = _objective(use_logh=True)
    m = np.log(fwd.h.function.x.array.copy())

    _, g_map = obj.value_and_grad(m)
    _, g_post = obj.value_and_grad(m, include_jacobian=True)

    np.testing.assert_allclose(g_post - g_map, -1.0, rtol=1e-12, atol=1e-12)


def test_direct_h_parametrization_also_matches_finite_differences():
    obj, fwd = _objective(use_logh=False, alpha=1e-6)
    h0 = fwd.h.function.x.array.copy()

    rng = np.random.default_rng(11)
    direction = rng.normal(size=h0.size)
    direction /= np.linalg.norm(direction)

    _, g = obj.value_and_grad(h0)
    analytic = float(g @ direction)

    eps = 1e-6
    numeric = (obj.value(h0 + eps * direction) - obj.value(h0 - eps * direction)) / (
        2.0 * eps
    )

    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-8)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_objective.py -v
```

Expected: all six FAIL with `ModuleNotFoundError: No module named 'objective'`.

- [ ] **Step 3: Write the module**

Create `src/objective.py`:

```python
# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# pde imports
from petsc4py import PETSc
import ufl

# dolfinx imports
from dolfinx import fem

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver


class Objective:
    r"""
    The regularized negative log-posterior and its gradient, as plain arrays.

        J(m) = \frac{1}{2\sigma^2}\lVert B T(h) - d\rVert^2
             + \frac{\alpha}{2}\int_\Omega \lvert\nabla h\rvert^2\,\mathrm{d}x

    with :math:`h = \exp(m)` when ``use_logh`` is set.

    This is the sampler-neutral boundary: it takes and returns NumPy arrays and
    holds no optimizer-specific state, so PETSc TAO and a BlackJAX ``custom_vjp``
    bridge can both consume it.

    Every global scalar is reduced across ranks. ``fem.assemble_scalar`` returns
    only the calling rank's contribution, so an unreduced value would make each
    rank minimize a different objective against a correctly-reduced gradient.

    Parameters
    ----------
    forward  : the forward solver, owning the shared coefficient h.
    adjoint  : the adjoint solver, owning the observation model and alpha.
    use_logh : optimize/sample in m = log h rather than in h directly.
    """

    def __init__(
        self,
        forward: SteadyHeat2DForwardSolver,
        adjoint: SteadyHeat2DAdjointSolver,
        use_logh: bool = True,
    ):
        self.fwd = forward
        self.adj = adjoint
        self.use_logh = use_logh
        self.comm = forward.disc.mesh.comm

        # Cache the regularization form once. Rebuilding it per evaluation costs
        # little at 10^3 optimizer iterations but is material at 10^4-10^6
        # leapfrog steps.
        h_func = self.fwd.h.function
        self._reg_form = fem.form(
            ufl.inner(ufl.grad(h_func), ufl.grad(h_func)) * ufl.dx
        )

    @property
    def n_parameters(self) -> int:
        """Length of the parameter vector this objective accepts."""
        return self.fwd.h.function.x.array.size

    def _write_parameters(self, m: np.ndarray) -> None:
        """Push the parameter vector into the shared coefficient."""
        values = np.asarray(m, dtype=np.float64)
        if values.size != self.n_parameters:
            raise ValueError(
                f"parameter vector has length {values.size}, "
                f"expected {self.n_parameters}."
            )
        h_array = np.exp(values) if self.use_logh else values
        self.fwd.h.function.x.array[:] = h_array
        self.fwd.h.function.x.scatter_forward()

    def _regularization(self) -> float:
        """Reduced ``alpha * integral |grad h|^2``."""
        local = fem.assemble_scalar(self._reg_form)
        return self.adj.alpha * self.comm.allreduce(local, op=MPI.SUM)

    def value(self, m: np.ndarray, include_jacobian: bool = False) -> float:
        r"""
        Evaluate :math:`J` at the parameter vector ``m``.

        Parameters
        ----------
        m                : parameter vector (``log h`` when ``use_logh``).
        include_jacobian : add the change-of-variables term for sampling in
                           ``log h``. Optimization does not need it; a posterior
                           density does.

        Returns
        -------
        float : the objective, identical on every MPI rank.
        """
        self._write_parameters(m)
        self.fwd.solve()

        misfit = self.adj.obs.neg_log_likelihood(self.fwd.T.x.array, self.adj.d)
        J = misfit + 0.5 * self._regularization()

        if include_jacobian and self.use_logh:
            # h = exp(m) => log|dh/dm| = sum(m); subtracting it turns the MAP
            # objective into the negative log-posterior in m.
            #
            # Sum ONLY the owned dofs before reducing: `m` spans owned+ghost, so
            # reducing the full local array would count every ghost twice.
            n_owned = self.fwd.disc.n_dofs_owned
            local_sum = float(np.sum(np.asarray(m, dtype=np.float64)[:n_owned]))
            J -= float(self.comm.allreduce(local_sum, op=MPI.SUM))
        return float(J)

    def value_and_grad(
        self, m: np.ndarray, include_jacobian: bool = False
    ) -> tuple[float, np.ndarray]:
        r"""
        Evaluate :math:`J` and :math:`\mathrm{d}J/\mathrm{d}m` together.

        One forward solve and one adjoint solve, regardless of the parameter count.

        Returns
        -------
        (J, gradient) : the objective and a ``(n_parameters,)`` array.
        """
        J = self.value(m, include_jacobian=include_jacobian)

        self.adj.solve()
        grad_vec = self.adj.assemble_gradient()
        grad_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        with grad_vec.localForm() as g_local:
            grad = np.array(g_local.array[: self.n_parameters], dtype=np.float64)

        if self.use_logh:
            # Chain rule: dJ/dm = dJ/dh * dh/dm = dJ/dh * h
            grad = grad * self.fwd.h.function.x.array[: self.n_parameters]
            if include_jacobian:
                grad = grad - 1.0

        return J, grad


def evaluate_objective(
    forward: SteadyHeat2DForwardSolver, adjoint: SteadyHeat2DAdjointSolver
) -> float:
    r"""
    Evaluate :math:`J` at the forward solver's *current* state.

    A convenience for tests that mutate ``h`` directly rather than through a
    parameter vector. Reduced across ranks like ``Objective.value``.
    """
    comm = forward.disc.mesh.comm
    misfit = adjoint.obs.neg_log_likelihood(forward.T.x.array, adjoint.d)
    h_func = forward.h.function
    local_reg = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(h_func), ufl.grad(h_func)) * ufl.dx)
    )
    reg = adjoint.alpha * comm.allreduce(local_reg, op=MPI.SUM)
    return float(misfit + 0.5 * reg)
```

- [ ] **Step 4: Register and install**

Add `"objective"` to `py-modules` in `pyproject.toml`, then `pip install -e . -q`.

- [ ] **Step 5: Point the gradient tests at the new objective**

In `tests/_helpers.py`, delete `eval_obj` entirely (it computed the superseded mass-weighted misfit without an MPI reduction). Keep `pick_random_test_direction`, `h_true`, `h0`, and `update_h`.

In each of `tests/test_grad_finitediff.py` and `tests/test_grad_taylorexp.py`, replace the `eval_obj` import and calls:

```python
from objective import evaluate_objective
```

and every `eval_obj(fwd, T_obs, noise_sigma, reg_alpha)` becomes `evaluate_objective(fwd, adj)`.

`tests/test_grad_forwarddiff.py` compares against the tangent-linear solve rather than the objective; update only its adjoint construction (done in Task 3, Step 5).

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest tests/test_objective.py -v
pytest -q -m gradcheck
```

Expected: 6 passed, then `3 passed`.

The gradient checks are now validating a *different* objective than before, so their numbers will differ from Tiers 1–2 — that is expected and correct. What must hold is that the adjoint gradient still matches finite differences and that the Taylor remainder rate is still ≈ 2. If the rate degrades, the adjoint source or the chain rule is wrong.

- [ ] **Step 7: Commit**

```bash
black src tests
git add src/objective.py tests/test_objective.py tests/_helpers.py tests/test_grad_finitediff.py tests/test_grad_taylorexp.py tests/test_grad_forwarddiff.py pyproject.toml
git commit -m "Add the pure MPI-reduced objective"
```

---

### Task 5: Reduce TAO to an adapter, and retire `add_noise`

With `Objective` owning evaluation, `_objgrad` becomes a translation layer between PETSc vectors and NumPy arrays. `add_noise` is removed: observations now live in sensor space, and leaving a DOF-space noise generator would invite exactly the mismatch this tier eliminates.

**Files:**
- Modify: `src/tao_solver.py`
- Modify: `src/forward_solver.py` (remove `add_noise`)
- Modify: `scripts/make_fixture.py`
- Modify: `tests/test_smoke.py`, `tests/test_tao_postconditions.py` (drop `add_noise` usage)

**Interfaces:**
- Consumes: `Objective` (Task 4).
- Produces: `SteadyHeat2DTAOSolver(forward, adjoint, ...)` unchanged externally, now delegating to `Objective`. `.objective` exposes the `Objective` for reuse by a sampler.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tao_postconditions.py` (replacing its existing `_solved_pair`, which used `add_noise`):

```python
def _sensor_grid(n_side=4):
    g = np.linspace(0.15, 0.85, n_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _solved_pair(nmesh=8):
    from sensors import SensorOperator
    from observation import GaussianSensorNoise

    truth = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    truth.solve()
    d = SensorOperator(truth.disc, _sensor_grid()).apply(truth.T.x.array)

    fwd = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    obs = GaussianSensorNoise(SensorOperator(fwd.disc, _sensor_grid()), sigma=1e-2)
    adj = SteadyHeat2DAdjointSolver(fwd, obs, d, alpha=5e-3)
    adj.solve()
    return fwd, adj


def test_tao_exposes_the_objective_for_reuse():
    """A sampler must be able to take the same objective TAO optimizes."""
    fwd, adj = _solved_pair()
    tao = SteadyHeat2DTAOSolver(fwd, adj, gatol=1e-8, grtol=1e-8, mit=20)

    from objective import Objective

    assert isinstance(tao.objective, Objective)
    assert tao.objective.n_parameters == fwd.h.function.x.array.size


def test_forward_solver_no_longer_exposes_add_noise():
    """Observations live in sensor space; a DOF-space generator would mismatch."""
    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0 + x[0], q=1.0)
    assert not hasattr(fwd, "add_noise")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_tao_postconditions.py -v
```

Expected: both new tests FAIL — `tao.objective` does not exist and `add_noise` is still present.

- [ ] **Step 3: Delegate `_objgrad` to the objective**

In `src/tao_solver.py`, add to the `# local imports` banner:

```python
from objective import Objective
```

Construct it in `__init__`, immediately after the `fem.Function` guard added in Tier 1:

```python
        self.objective = Objective(forward, adjoint, use_logh=use_logh)
```

Replace the body of `_objgrad`:

```python
    def _objgrad(self, tao, X: PETSc.Vec, G: PETSc.Vec):
        """
        PETSc TAO callback: translate between PETSc vectors and the Objective.

        Parameters
        ----------
        tao : PETSc.TAO
        X   : the current parameter vector (log h when use_logh).
        G   : output vector receiving dJ/dm.

        Returns
        -------
        J : the objective value, identical on every rank.
        """
        X.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with X.localForm() as x_local:
            m = np.array(x_local.array, dtype=np.float64)

        J, grad = self.objective.value_and_grad(m)

        with G.localForm() as g_local:
            g_local.array[: grad.size] = grad
        G.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        if self.verbose >= 2 and MPI.COMM_WORLD.rank == 0:
            print("Current J =", J)
        if self.verbose == 3 and MPI.COMM_WORLD.rank == 0:
            print("Current |G| =", G.norm())
        return J
```

Delete `self.alpha = adjoint.alpha` and `self.sigma2 = adjoint.sigma2` from `__init__` — snapshotting them let `J` and `G` desynchronize if the adjoint's values were later mutated. The objective reads them live from the adjoint.

- [ ] **Step 4: Remove `add_noise`**

Delete the entire `add_noise` method from `src/forward_solver.py`, and delete the `noiseless` branch of `plot_output_temperature` that reads `self.T_obs`, simplifying it to plot `self.T` only:

```python
    def plot_output_temperature(self, zero_point: float = 300.0, **kwargs):
        """
        Plot the temperature distribution on a pyvista.UnstructuredGrid.

        Parameters
        ----------
        zero_point : reference temperature subtracted from T(x,y).
        **kwargs   : forwarded to `plotting_utils.plot_scalar_mesh()`.
        """
        if not hasattr(self, "T"):
            raise RuntimeError("No solution available. Call solve() first.")
        vals = self.T.x.array[: self.disc.n_dofs_local] - zero_point
```

Keep the rest of the method (the `zero_point` branch and the two `plot_scalar_mesh` calls) unchanged.

Also remove `if hasattr(self, "T_obs"): xdmf.write_function(self.T_obs)` from `export_xdmf` — there is no `T_obs` any more.

- [ ] **Step 5: Emit sensor observations from the fixture script**

In `scripts/make_fixture.py`, replace the `fwd.add_noise(...)` call and extend the metadata:

```python
    # Sensor grid: fixed physical locations, independent of the mesh.
    g = np.linspace(0.05, 0.95, args.n_sensors_side)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    sensor_points = np.column_stack([xx.ravel(), yy.ravel()])

    sensors = SensorOperator(fwd.disc, sensor_points)
    observation = GaussianSensorNoise(sensors, sigma=args.sigma)
    d = observation.sample(fwd.T.x.array, seed=args.seed)
```

Add the CLI flag alongside the others:

```python
    parser.add_argument("--n-sensors-side", type=int, default=16, dest="n_sensors_side")
```

Add the imports to the `# local imports` banner:

```python
from sensors import SensorOperator
from observation import GaussianSensorNoise
```

And write the data next to the field, replacing the metadata block's digest computation:

```python
    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            args.out.with_suffix(".sensors.npz"),
            points=sensor_points,
            d=d,
            sigma=args.sigma,
            seed=args.seed,
        )
        digest = hashlib.sha256(
            np.ascontiguousarray(d, dtype=np.float64).tobytes()
        ).hexdigest()
        meta = {
            "seed": args.seed,
            "sigma": args.sigma,
            "nmesh": args.nmesh,
            "n_sensors": int(sensors.n_sensors),
            "q": args.q,
            "DBC_value": args.dbc,
            "h_true": "1 + 6*x^2 + x/(1 + 2*y^2)",
            "observations_sha256": digest,
        }
```

- [ ] **Step 6: Drop `add_noise` from the smoke tests**

In `tests/test_smoke.py`, delete any call to `fwd.add_noise(...)` and the `noiseless=False` plotting assertion, keeping the rest.

- [ ] **Step 7: Run the full suite**

```bash
pytest -q
```

Expected: all tests pass. Record the count.

- [ ] **Step 8: Verify the fixture regenerates**

```bash
python scripts/make_fixture.py --seed 0 --sigma 1e-3 --nmesh 64 --out /tmp/fx.xdmf
python -c "
import json, numpy as np
meta = json.load(open('/tmp/fx.meta.json')); print(meta)
z = np.load('/tmp/fx.sensors.npz')
print('sensor points:', z['points'].shape, 'observations:', z['d'].shape)
assert z['d'].shape[0] == meta['n_sensors']
"
```

Expected: the metadata reports `n_sensors`, and `d` has one value per sensor.

- [ ] **Step 9: Commit**

```bash
black src tests scripts
git add src/tao_solver.py src/forward_solver.py scripts/make_fixture.py tests/test_smoke.py tests/test_tao_postconditions.py
git commit -m "Delegate the TAO callback to the objective and retire add_noise"
```

---

### Task 6: Prove mesh independence and MPI correctness

Two properties justify this tier's cost. Neither is checked by any test written so far, and both are the reason the sensor model was chosen over the alternatives.

**Files:**
- Create: `tests/test_mesh_independence.py`
- Modify: `.github/workflows/run-gradchecks.yml` (add a 2-rank MPI job)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: the tier's exit evidence.

- [ ] **Step 1: Write the mesh-independence test**

Create `tests/test_mesh_independence.py`:

```python
# tests/test_mesh_independence.py
# numerical imports
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from sensors import SensorOperator
from observation import GaussianSensorNoise
from objective import Objective


SENSOR_SIDE = 4


def _sensor_grid():
    g = np.linspace(0.15, 0.85, SENSOR_SIDE)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _misfit_at_truth(nmesh, d):
    """Evaluate the data misfit at the true conductivity on an nmesh grid."""
    fwd = SteadyHeat2DForwardSolver(
        nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    obs = GaussianSensorNoise(
        SensorOperator(fwd.disc, _sensor_grid()), sigma=1e-2
    )
    return obs.neg_log_likelihood(fwd.T.x.array, d), fwd


def test_data_vector_length_is_independent_of_the_mesh():
    """The property that makes multi-fidelity comparison meaningful."""
    lengths = []
    for nmesh in (8, 16, 32):
        fwd = SteadyHeat2DForwardSolver(
            nmesh=nmesh, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
        )
        fwd.solve()
        B = SensorOperator(fwd.disc, _sensor_grid())
        lengths.append(B.apply(fwd.T.x.array).size)

    assert lengths == [SENSOR_SIDE**2] * 3, (
        f"data length varied with the mesh: {lengths}; coarse and fine fidelities "
        "would then describe different statistical models"
    )


def test_misfit_converges_under_refinement():
    """Refining must improve the prediction of fixed data, not inflate the misfit.

    Under the old all-dof model the misfit grew with N_dof; here it must settle.
    """
    # Generate data once on a fine mesh, then evaluate the misfit on coarser ones.
    fine = SteadyHeat2DForwardSolver(
        nmesh=64, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fine.solve()
    d = SensorOperator(fine.disc, _sensor_grid()).apply(fine.T.x.array)

    misfits = [_misfit_at_truth(nmesh, d)[0] for nmesh in (8, 16, 32)]

    # Monotone decrease toward zero as the forward model resolves the true field.
    assert misfits[1] < misfits[0]
    assert misfits[2] < misfits[1]
    assert misfits[2] < 0.05 * misfits[0]


def test_gradient_dimension_tracks_parameters_not_data():
    obj_dims = []
    for nmesh in (8, 16):
        fwd = SteadyHeat2DForwardSolver(
            nmesh=nmesh, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0, DBC_value=300.0
        )
        fwd.solve()
        B = SensorOperator(fwd.disc, _sensor_grid())
        obs = GaussianSensorNoise(B, sigma=1e-2)
        d = np.zeros(B.n_sensors)
        adj = SteadyHeat2DAdjointSolver(fwd, obs, d, alpha=1e-6)
        obj = Objective(fwd, adj, use_logh=True)

        m = np.log(fwd.h.function.x.array.copy())
        _, g = obj.value_and_grad(m)
        obj_dims.append(g.size)

    # The parameter dimension grows with the mesh; the data dimension does not.
    assert obj_dims[1] > obj_dims[0]
```

- [ ] **Step 2: Run it**

```bash
pytest tests/test_mesh_independence.py -v
```

Expected: 3 passed. If `test_misfit_converges_under_refinement` fails, the sensor interpolation or the forward model is not converging — investigate rather than loosening the factor.

- [ ] **Step 3: Write the MPI reduction test**

Create `tests/mpi_check_objective.py` (not collected by pytest — it is launched under `mpirun`):

```python
"""Verify the objective is identical on every rank. Run under mpirun."""

# numerical imports
import sys

import numpy as np

sys.path.insert(0, "src")

# mpi imports
from mpi4py import MPI

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from sensors import SensorOperator
from observation import GaussianSensorNoise
from objective import Objective


def main():
    comm = MPI.COMM_WORLD
    g = np.linspace(0.15, 0.85, 4)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    fwd = SteadyHeat2DForwardSolver(
        nmesh=16, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0, DBC_value=300.0
    )
    fwd.solve()
    B = SensorOperator(fwd.disc, pts)
    obs = GaussianSensorNoise(B, sigma=1e-2)
    d = np.zeros(B.n_sensors)
    adj = SteadyHeat2DAdjointSolver(fwd, obs, d, alpha=1e-3)
    obj = Objective(fwd, adj, use_logh=True)

    m = np.log(fwd.h.function.x.array.copy())
    J, _ = obj.value_and_grad(m)

    gathered = comm.allgather(J)
    if comm.rank == 0:
        spread = max(gathered) - min(gathered)
        print("J per rank:", gathered)
        print("spread:", spread)
        assert spread < 1e-12, (
            "the objective differs across ranks: it is not MPI-reduced, so each "
            "rank would minimize a different function"
        )
        print("OK: objective is identical on all ranks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it on two ranks**

```bash
mpirun -n 2 python tests/mpi_check_objective.py
```

Expected: `OK: objective is identical on all ranks`, with a spread of `0.0`.

This is the check that would have caught the original bug: before this tier, `fem.assemble_scalar` returned 0.5 where the true value was 1.0 on a 2-rank split.

- [ ] **Step 5: Add the MPI job to CI**

In `.github/workflows/run-gradchecks.yml`, add a third job:

```yaml
  mpi:
    name: MPI consistency (2 ranks)
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: mamba-org/setup-micromamba@v1
        with:
          environment-file: environment.yml
          environment-name: steady-2d-heat-inverse
          cache-environment: true
      - name: Install package
        shell: bash -l {0}
        run: pip install -e ".[dev]"
      - name: Objective must be rank-independent
        shell: bash -l {0}
        run: mpirun -n 2 --oversubscribe python tests/mpi_check_objective.py
```

- [ ] **Step 6: Commit and push**

```bash
black tests
git add tests/test_mesh_independence.py tests/mpi_check_objective.py .github/workflows/run-gradchecks.yml
git commit -m "Prove mesh independence and MPI consistency of the objective"
git push origin master
```

---

## Tier 3 Verification Round

- [ ] **Full suite passes**

```bash
pytest -q
```
Expected: all tests pass. Record the count for the handoff note.

- [ ] **All three gradient checks hold against the new objective**

```bash
pytest -q -m gradcheck
```
Expected: `3 passed`. Values differ from Tiers 1–2 — the objective changed — but finite differences, the tangent-linear solve, and the Taylor remainder must all still agree with the adjoint, and the remainder rate must remain ≈ 2.

- [ ] **The objective is rank-independent**

```bash
mpirun -n 2 python tests/mpi_check_objective.py
```
Expected: `OK: objective is identical on all ranks`.

- [ ] **Data dimension is mesh-independent**

```bash
pytest -q tests/test_mesh_independence.py -v
```
Expected: 3 passed — the property that makes multi-fidelity meaningful.

- [ ] **The objective is reusable outside TAO**

This is the deliverable for the BlackJAX follow-on. Confirm it works standalone:

```bash
python -c "
import sys; sys.path.insert(0,'src')
import numpy as np
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from sensors import SensorOperator
from observation import GaussianSensorNoise
from objective import Objective

g = np.linspace(0.15,0.85,4); xx,yy = np.meshgrid(g,g,indexing='ij')
pts = np.column_stack([xx.ravel(), yy.ravel()])
fwd = SteadyHeat2DForwardSolver(nmesh=16, h=lambda x: 2.0+3.0*x[0]**2, q=1.0, DBC_value=300.0)
fwd.solve()
B = SensorOperator(fwd.disc, pts); obs = GaussianSensorNoise(B, sigma=1e-2)
adj = SteadyHeat2DAdjointSolver(fwd, obs, np.zeros(B.n_sensors), alpha=1e-3)
obj = Objective(fwd, adj, use_logh=True)

m = np.log(fwd.h.function.x.array.copy())
J, grad = obj.value_and_grad(m, include_jacobian=True)
print('log-posterior:', J)
print('gradient shape:', grad.shape, 'finite:', bool(np.isfinite(grad).all()))
print('no PETSc or TAO object required')
"
```
Expected: a finite value and gradient, with no optimizer involved.

- [ ] **Docs build clean**

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: `build succeeded.`

- [ ] **CI green, including the new MPI job**

```bash
gh run list --limit 6
```
Expected: no `failure`.

**Exit criteria:** every box checked.

## Handoff: what this unlocks

- **BlackJAX/NUTS.** `Objective.value_and_grad(m, include_jacobian=True)` is the negative log-posterior and its gradient in the unconstrained parameter `m`. Wrap it with `jax.pure_callback` plus `jax.custom_vjp` — the exact gradient already exists, which is the part that is normally hard — and pass the result as BlackJAX's `logdensity_fn`. No JAX port of the FEM stack is required.
- **Multi-fidelity.** Construct two `Discretization`s at different `nmesh` with the *same* sensor points and the *same* `d`. Both `Objective`s then estimate the same quantity, which is the precondition for control variates or a multilevel telescoping estimator.
- **Documentation.** `docs/theory.md` and `docs/usage.md` still describe the mass-weighted misfit, the DOF-space `T_obs`, and `add_noise`. They are now wrong. Re-syncing them is the first task of whatever plan follows this tier — the same scoping mistake that produced Tier 1's docs debt would otherwise repeat.
