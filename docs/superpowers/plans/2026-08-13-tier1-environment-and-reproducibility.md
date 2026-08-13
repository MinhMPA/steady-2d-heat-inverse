# Tier 1: Environment, CI, and Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a pinned, reproducible environment on DOLFINx 0.11 with green CI, decouple the compute path from the plotting stack, and replace the non-reproducible sweep artifacts with a deterministic seeded fixture.

**Architecture:** This tier touches only environment, CI, import structure, and solver *construction* — never the objective, the noise model, or the class hierarchy. That boundary is deliberate: Tier 2 restructures the classes and Tier 3 rewrites the objective, so nothing here will be redone later.

**Tech Stack:** conda-forge (DOLFINx 0.11 / PETSc / MPI), pytest, GitHub Actions, Sphinx.

## Global Constraints

- **Run every command in the new environment, from Task 1 Step 3 onward.** `conda activate` does not persist between shell invocations in this harness — each command starts a fresh shell that auto-activates the *old* `steady-2d-heat-inverse_py313forge` (DOLFINx 0.9.0). Prefix commands with `conda run -n s2dhi-011 ...`, or you will verify against 0.9.0 while believing you are on 0.11. This applies to every `pytest`, `python`, `black`, and `sphinx-build` invocation in Tasks 2–6, whose command blocks are written without the prefix for readability.
- **The core environment has no `pyvista`.** It is an optional extra (`pip install -e ".[plot]"`), because pyvista/vtk conflict with fenics-dolfinx 0.11 on `libboost`. Task 1 installs `[dev,plot]` locally so the suite can run before Task 2 makes the import lazy; CI installs only `[dev]`, which is what proves the compute path is genuinely plotting-free.
- **Import convention:** sources install as *flat top-level modules* (`pyproject.toml` `py-modules`). Always `from forward_solver import ...`, never `from src.forward_solver import ...`.
- **Test imports:** `tests/` is a package (`tests/__init__.py` exists). Import shared helpers relatively (`from ._helpers import ...`); import code under test flatly.
- **Formatting:** `black`. Run `black src tests scripts` before each commit.
- **Import banners:** group imports under the existing comment banners (`# type imports`, `# numerical imports`, `# mpi imports`, `# pde imports`, `# dolfinx imports`, `# local imports`).
- **Docstring reST rule:** an indented equation under a `:`-terminated lead-in must be followed by a blank line before the next unindented paragraph, or `sphinx-build -W` fails.
- **Commits:** short imperative subject lines, no `feat:`/`fix:` prefixes.
- **New tests must be fast** (< ~5 s each) and must NOT carry `pytest.mark.gradcheck`. Use `nmesh=4` or `nmesh=8`.
- **Do not touch** `notebooks/EvaluateSolution.ipynb`, `notebooks/solution.vtu` — uncommitted user work. Never `git add -A`.
- **Do not modify** the objective (`_objgrad`), the noise model (`add_noise`), or the class hierarchy. Those belong to Tiers 2 and 3.

## Out of Scope (owned by later tiers)

| Item | Owner |
|---|---|
| MPI reduction of `J` | Tier 3 (the new objective module is reduced by construction) |
| `add_noise` per-rank `Bcast` bug | Tier 3 (`add_noise` is replaced by the sensor model) |
| Adjoint/forward inheritance | Tier 2 |
| `mesh.geometry.x.shape[0]` as DOF count | Tier 2 |
| Form caching in the objective | Tier 3 |

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `environment.yml` | modify | fully pinned conda spec on DOLFINx 0.11 |
| `src/forward_solver.py` | modify | `petsc_options_prefix`, convergence-checking PETSc defaults |
| `src/adjoint_solver.py` | modify | `petsc_options_prefix` |
| `src/tao_solver.py` | modify | re-sync `fwd.h.function` to the optimum after `solve()` |
| `src/plotting_utils.py` | modify | lazy `pyvista` import |
| `scripts/make_fixture.py` | create | deterministic seeded observation fixture |
| `tests/test_smoke.py` | create | construct-and-solve smoke test; lazy-import test; convergence test |
| `tests/test_tao_postconditions.py` | create | shared `h` matches the returned optimum |
| `.github/workflows/run-gradchecks.yml` | modify | add a docs-build job |
| `.gitignore` | modify | ignore `notebooks/*.npy` |

---

### Task 1: Pin the environment and migrate to DOLFINx 0.11

The stack is unpinned. CI resolved to Python 3.14 with a VTK where `vtkCapsuleSource` moved, breaking `import pyvista` and therefore every module. Separately, DOLFINx 0.11 makes `petsc_options_prefix` a mandatory keyword-only argument to `LinearProblem`, so a fresh environment would fail on first solver construction.

**Files:**
- Modify: `environment.yml`
- Modify: `src/forward_solver.py` (the `LinearProblem(...)` call)
- Modify: `src/adjoint_solver.py` (the `LinearProblem(...)` call)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a working DOLFINx 0.11 environment. Every later task in every tier assumes `petsc_options_prefix` is present on both solvers.

- [ ] **Step 1: Record the current working versions**

Before changing anything, capture what works today so you can roll back:

```bash
conda list -n steady-2d-heat-inverse --export > /tmp/env-before-tier1.txt
python -c "import dolfinx, petsc4py, mpi4py; print(dolfinx.__version__, petsc4py.__version__, mpi4py.__version__)"
```

Expected: `0.9.0 3.23.5 4.1.0` (or whatever your machine reports — record it).

- [ ] **Step 2: Write the pinned environment file**

The core environment is **compute-only**: `pyvista` and `vtk` are deliberately absent.

This is not an oversight. `pyvista 0.45`/`vtk 9.4` cannot co-exist with `fenics-dolfinx 0.11` — they conflict on `libboost`, and the solver only succeeds if they float to `0.48.4`/`9.6.2`. Rather than drag a two-minor-version plotting upgrade into every compute environment and every CI job, rendering becomes an optional extra (Step 2b). Task 2 makes the `pyvista` import lazy, so the library is fully usable without it.

Replace `environment.yml` entirely:

```yaml
name: steady-2d-heat
channels:
  - conda-forge
dependencies:
  - python=3.12.*
  - numpy=2.*
  - scipy=1.*
  - pandas=2.*
  - mpi4py=4.*
  - petsc4py=3.25.*
  - fenics-basix=0.11.*
  - fenics-ufl=2026.1.*
  - fenics-dolfinx=0.11.*
  - h5py=3.*
  - jupyterlab
  - pip:
      - -e .
```

These pins are **verified to solve** on conda-forge/osx-arm64 via `mamba create --dry-run`. Do not substitute `fenics-ufl=2025.*` or `petsc4py=3.23.*` — those are the 0.9.0-era versions and will fail with `fenics-dolfinx 0.11 requires fenics-ufl =2026.1`.

- [ ] **Step 2b: Declare plotting as an optional extra**

In `pyproject.toml`, extend the optional dependencies:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-xdist>=3.5",
]
plot = [
  "pyvista>=0.48",
]
```

Anyone who wants figures installs `pip install -e ".[plot]"`; note that doing so inside a DOLFINx 0.11 environment will resolve `vtk` to 9.6.x, which is expected.

- [ ] **Step 3: Build the new environment under a different name**

Do **not** overwrite your working environment — build alongside it so rollback is trivial:

Use `mamba`, not `conda` — a conda solve of the full FEniCSx stack is slow enough to risk a command timeout:

```bash
mamba env create -n s2dhi-011 -f environment.yml
conda run -n s2dhi-011 pip install -e ".[dev]"
conda run -n s2dhi-011 python -c "import dolfinx; print('dolfinx', dolfinx.__version__)"
```

Expected: `dolfinx 0.11.0`.

**`conda activate` does not persist between shell invocations here** — each command starts a fresh shell that auto-activates the *old* environment. Prefix every subsequent command with `conda run -n s2dhi-011`, or you will silently verify against DOLFINx 0.9.0 and believe you are on 0.11.

If mamba cannot solve the spec, report BLOCKED with the solver output rather than loosening pins silently — an unpinned environment is the bug being fixed. (The pins in Step 2 have already been dry-run verified, so a failure here means something changed upstream and is worth surfacing.)

- [ ] **Step 4: Confirm the API break is real before fixing it**

```bash
conda run -n s2dhi-011 python -c "
import sys; sys.path.insert(0,'src')
from forward_solver import SteadyHeat2DForwardSolver
SteadyHeat2DForwardSolver(nmesh=4, h=1.0, q=1.0)
"
```

Expected: `TypeError` naming `petsc_options_prefix` as a missing required argument. This is the RED state.

Two things this step also proves incidentally: that the module imports at all with no `pyvista` installed (Task 2 makes that robust; here it may still fail, which is fine and expected at this point), and that you are genuinely running against 0.11. If you instead see the solver construct successfully, you are in the wrong environment — check the `conda run` prefix.

- [ ] **Step 5: Add the prefix to the forward solver**

In `src/forward_solver.py`, replace the `LinearProblem` construction:

```python
        self.T = fem.Function(self.V, name="Temperature")
        self.problem = LinearProblem(
            self.a,
            self.L,
            u=self.T,
            bcs=self.bcs,
            petsc_options_prefix="s2dhi_fwd_",
            petsc_options=self.petsc_opts,
        )
```

- [ ] **Step 6: Add the prefix to the adjoint solver**

In `src/adjoint_solver.py`, replace the `LinearProblem` construction:

```python
        self.lambda_L = fem.Function(self.V, name="AdjointState")
        self.problem = LinearProblem(
            self.a,
            self.L,
            u=self.lambda_L,
            bcs=self.bcs,
            petsc_options_prefix="s2dhi_adj_",
            petsc_options=opts,
        )
```

The two prefixes must differ — PETSc uses them to namespace options, and a collision would silently apply the forward solver's options to the adjoint.

- [ ] **Step 7: Verify construction and the full suite**

**Install the plotting extra for this step.** Until Task 2 makes the import lazy, `plotting_utils` imports `pyvista` at module scope, and `forward_solver` and `domain_coefficient` both import from it — so with no `pyvista` present, *every* test fails at collection. Task 2 removes this requirement; CI (Task 6) will install only `[dev]`.

```bash
conda run -n s2dhi-011 pip install -e ".[dev,plot]"
conda run -n s2dhi-011 python -c "
import sys; sys.path.insert(0,'src')
from forward_solver import SteadyHeat2DForwardSolver
f = SteadyHeat2DForwardSolver(nmesh=4, h=1.0, q=1.0); f.solve()
print('construct+solve OK')
"
conda run -n s2dhi-011 pytest -q
```

Expected: `construct+solve OK`, then `16 passed`.

Record the resolved `pyvista` and `vtk` versions — they will be 0.48.x and 9.6.x, and Task 2's plotting check needs to know rendering still works on them:

```bash
conda run -n s2dhi-011 python -c "import pyvista, vtk; print('pyvista', pyvista.__version__, '| vtk', vtk.VTK_VERSION)"
```

If any gradient check now fails, the PETSc upgrade shifted tolerances. Report the actual numbers rather than loosening `rtol` — a genuine gradient regression and a tolerance shift look identical from the assertion alone, and only the numbers distinguish them.

- [ ] **Step 8: Commit**

```bash
black src
git add environment.yml pyproject.toml src/forward_solver.py src/adjoint_solver.py
git commit -m "Pin a compute-only DOLFINx 0.11 stack and make plotting an extra"
```

---

### Task 2: Make the compute path importable without the plotting stack

`src/plotting_utils.py` imports `pyvista` at module scope, and both `forward_solver.py` and `domain_coefficient.py` import from it. A broken VTK therefore makes the *entire library* unimportable — which is precisely why `install-and-import.yml` failed, not just the test workflow. Rendering is optional; compute is not.

**Files:**
- Modify: `src/plotting_utils.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: Task 1's working environment.
- Produces: `tests/test_smoke.py`, extended by Tasks 3 and 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke.py`:

```python
# tests/test_smoke.py
# numerical imports
import subprocess
import sys

import numpy as np
import pytest


def test_importing_forward_solver_does_not_import_pyvista():
    """Compute must not depend on the rendering stack.

    Run in a subprocess: pytest itself may already have imported pyvista via
    another test module, which would mask the coupling.
    """
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import forward_solver;"
        "print('pyvista' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", (
        "importing forward_solver pulled in pyvista; "
        "a broken VTK would make the whole library unimportable"
    )


def test_forward_solver_constructs_and_solves():
    """Catches constructor-signature breaks (e.g. petsc_options_prefix) in CI."""
    sys.path.insert(0, "src")
    from forward_solver import SteadyHeat2DForwardSolver

    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0 + x[0], q=1.0)
    T = fwd.solve()

    assert np.isfinite(T.x.array).all()
    assert T.x.array.size > 0
```

- [ ] **Step 2: Run the tests to verify the first one fails**

```bash
pytest tests/test_smoke.py -v
```

Expected: `test_importing_forward_solver_does_not_import_pyvista` FAILS (stdout is `True`).
`test_forward_solver_constructs_and_solves` PASSES already — it is a regression guard for Task 1, not a bug detector.

- [ ] **Step 3: Move the pyvista import inside the function**

In `src/plotting_utils.py`, delete the module-level `import pyvista as pv` and import it where it is used:

```python
# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# dolfinx imports
from dolfinx.plot import vtk_mesh

# type imports
from typing import Any
from numpy.typing import ArrayLike


def plot_scalar_mesh(
    mesh,
    data: ArrayLike,
    name: str,
    cmap: str = "viridis",
    show_edges: bool = False,
    n_labels: int = 5,
    user_scalar_bar: dict | None = None,
    return_plotter: bool = False,
    **mesh_kwargs: Any,
):
```

Then, as the first statement inside the function body — after the docstring and before the rank guard:

```python
    # Imported lazily so that compute-only paths stay importable without VTK.
    # `pyvista` is an optional extra: `pip install -e ".[plot]"`.
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ImportError(
            "Plotting requires the optional 'plot' extra, which is deliberately "
            "absent from the compute environment because pyvista/vtk conflict with "
            "fenics-dolfinx 0.11. Install it with: pip install -e \".[plot]\""
        ) from exc

    if MPI.COMM_WORLD.rank != 0:
        return
```

The wrapped `ImportError` matters: without it, a user who never installed the extra gets a bare `ModuleNotFoundError: No module named 'pyvista'` from deep inside a plotting helper, with no hint that the fix is an install extra rather than a broken environment.

Leave the rest of the function unchanged. Remove the now-unused `Sequence` import if present.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_smoke.py -v
pytest -q
```

Expected: 2 passed in the smoke module; `18 passed` overall.

- [ ] **Step 5: Confirm plotting still works**

The lazy import must not break rendering:

```bash
PYVISTA_OFF_SCREEN=true python -c "
import sys; sys.path.insert(0,'src')
from forward_solver import SteadyHeat2DForwardSolver
f = SteadyHeat2DForwardSolver(nmesh=4, h=lambda x: 1.0+x[0], q=1.0); f.solve()
g = f.plot_output_temperature(zero_point=0.0)
print('plot returned:', type(g).__name__)
"
```

Expected: `plot returned: UnstructuredGrid`.

- [ ] **Step 6: Commit**

```bash
black src tests
git add src/plotting_utils.py tests/test_smoke.py
git commit -m "Import pyvista lazily so compute paths need no VTK"
```

---

### Task 3: Fail loudly on unconverged linear solves

`LinearProblem` does not raise when KSP fails to converge, and neither solver checks `getConvergedReason()`. An unconverged forward or adjoint state currently flows into the objective and gradient silently. Under HMC that corrupts acceptance decisions; under optimization it produces a wrong gradient with no signal.

**Files:**
- Modify: `src/forward_solver.py` (the `default_opts` dict)
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `tests/test_smoke.py` from Task 2.
- Produces: `ksp_error_if_not_converged` in the default PETSc options, inherited by the adjoint solver (which copies `forward.petsc_opts`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_smoke.py`:

```python
def test_unconverged_solve_raises():
    """A KSP that cannot converge must raise, not return a wrong answer."""
    sys.path.insert(0, "src")
    from forward_solver import SteadyHeat2DForwardSolver

    # One iteration at an unreachable tolerance: convergence is impossible.
    fwd = SteadyHeat2DForwardSolver(
        nmesh=8,
        h=lambda x: 1.0 + x[0],
        q=1.0,
        petsc_opts={"ksp_max_it": 1, "ksp_rtol": 1e-14, "pc_type": "none"},
    )

    with pytest.raises(Exception) as excinfo:
        fwd.solve()

    assert "converge" in str(excinfo.value).lower() or "DIVERGED" in str(excinfo.value)


def test_default_options_enable_convergence_errors():
    sys.path.insert(0, "src")
    from forward_solver import SteadyHeat2DForwardSolver

    fwd = SteadyHeat2DForwardSolver(nmesh=4, h=1.0, q=1.0)
    assert fwd.petsc_opts["ksp_error_if_not_converged"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_smoke.py -v -k "converge"
```

Expected: both FAIL — `test_unconverged_solve_raises` because `solve()` returns normally, and `test_default_options_enable_convergence_errors` with `KeyError`.

- [ ] **Step 3: Enable convergence errors by default**

In `src/forward_solver.py`, replace the default PETSc options:

```python
        # Specify options for the PETSc KSP linear system solver.
        ## Default options are set to use the conjugate gradient method with hypre preconditioner.
        ## `ksp_error_if_not_converged` makes a failed solve raise instead of silently
        ## returning an unconverged state into the objective and gradient.
        default_opts = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1e-10,
            "ksp_error_if_not_converged": True,
        }
        self.petsc_opts = default_opts | (petsc_opts or {})
```

The adjoint solver inherits this: it copies `forward.petsc_opts` when `petsc_opts is None`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_smoke.py -v
pytest -q
```

Expected: 4 passed in the smoke module; `20 passed` overall.

If a gradient check now fails because a legitimate solve was marginally unconverged, that is a real finding — report it rather than reverting the option.

- [ ] **Step 5: Commit**

```bash
black src tests
git add src/forward_solver.py tests/test_smoke.py
git commit -m "Raise on unconverged KSP solves"
```

---

### Task 4: Re-sync the shared conductivity to the optimum after `solve()`

`_update_h` runs only inside the `_objgrad` callback, so after `tao.solve()` returns, the shared `fwd.h.function` holds whatever trial point TAO probed last — which for a rejected line-search step is not the optimum. `solve()` returns the correct array, but any subsequent `fwd.export_xdmf()`, `fwd.solve()`, or `h.plot_input_thermal_conductivity()` silently uses a non-optimal `h`.

**Files:**
- Modify: `src/tao_solver.py` (`solve`)
- Create: `tests/test_tao_postconditions.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the postcondition that after `solve()`, `fwd.h.function.x.array` equals the returned optimum. Tier 3's objective module relies on this when it re-evaluates at the solution.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tao_postconditions.py`:

```python
# tests/test_tao_postconditions.py
# numerical imports
import sys

import numpy as np

sys.path.insert(0, "src")

# local imports
from forward_solver import SteadyHeat2DForwardSolver
from adjoint_solver import SteadyHeat2DAdjointSolver
from tao_solver import SteadyHeat2DTAOSolver


def _solved_pair():
    """A tiny converged forward/adjoint pair with a spatially varying h."""
    truth = SteadyHeat2DForwardSolver(
        nmesh=8, h=lambda x: 1.0 + 6.0 * x[0] ** 2, q=1.0
    )
    truth.solve()
    T_obs = truth.add_noise(0.0, 1e-3, seed=0)

    fwd = SteadyHeat2DForwardSolver(
        nmesh=8, h=lambda x: 2.0 + 3.0 * x[0] ** 2, q=1.0
    )
    fwd.solve()
    adj = SteadyHeat2DAdjointSolver(
        fwd, T_obs.x.array, sigma=1e-3, alpha=5e-3, DBC_value=0.0
    )
    adj.solve()
    return fwd, adj


def test_shared_h_matches_returned_optimum_after_solve():
    """After solve(), the shared coefficient must be the optimum, not the last probe."""
    fwd, adj = _solved_pair()
    tao = SteadyHeat2DTAOSolver(fwd, adj, gatol=1e-8, grtol=1e-8, mit=50)

    sol = tao.solve()
    shared = fwd.h.function.x.array[: sol.size]

    np.testing.assert_allclose(shared, sol, rtol=1e-12, atol=0.0)


def test_solution_is_a_copy_not_a_live_view():
    """tao.solution must not alias TAO's internal vector."""
    fwd, adj = _solved_pair()
    tao = SteadyHeat2DTAOSolver(fwd, adj, gatol=1e-8, grtol=1e-8, mit=50)

    sol = tao.solve()
    before = sol.copy()
    tao.tao.getSolution().set(0.0)  # scribble on TAO's vector

    np.testing.assert_allclose(sol, before, rtol=0.0, atol=0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_tao_postconditions.py -v
```

Expected: at least `test_solution_is_a_copy_not_a_live_view` FAILS (the returned array aliases TAO's vector). `test_shared_h_matches_returned_optimum_after_solve` may pass or fail depending on whether TAO's final probe happened to be the accepted point — if it passes, it is still required as a regression guard, because the current code does not *guarantee* it. Record which you observed.

- [ ] **Step 3: Re-sync and copy in `solve()`**

In `src/tao_solver.py`, replace the body of `solve`:

```python
    def solve(self):
        """
        Solve for the thermal conductivity h(x,y) to minimize the objective function J.

        Returns
        -------
        numpy.ndarray : the solution vector for thermal conductivity h(x,y).
        """
        self.tao.solve(x=self.X0)
        if self.verbose > 0:
            print("Convergence Reason:", self.tao.getConvergedReason())
            print(
                "For more details, refer to https://petsc.org/release/manualpages/Tao/TaoConvergedReason/"
            )

        # TAO's last callback may have been a rejected trial point, so the shared
        # coefficient is not necessarily the optimum. Write the optimum back before
        # returning, so export/plot/re-solve all see the converged h.
        X_opt = self.tao.getSolution()
        self._update_h(X_opt)

        if self.use_logh:
            self.solution = np.exp(X_opt.array.copy())
        else:
            self.solution = X_opt.array.copy()
        return self.solution
```

`.copy()` detaches the returned array from TAO's vector, which is invalidated by later TAO state changes.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_tao_postconditions.py -v
pytest -q
```

Expected: 2 passed in the postconditions module; `22 passed` overall.

- [ ] **Step 5: Commit**

```bash
black src tests
git add src/tao_solver.py tests/test_tao_postconditions.py
git commit -m "Re-sync the shared conductivity to the optimum after solve"
```

---

### Task 5: Deterministic observation fixture, and retire the invalid sweep artifacts

`notebooks/ForwardSolve.ipynb` calls `add_noise(0., sigma)` with no seed, so `test_data/blackbox_output.h5` changes on every re-run — which is why it has shown as modified indefinitely. Separately, 10 of the 14 `notebooks/hsol_*.npy` files contain negative values and are therefore relative errors, not conductivity: they cannot be optimizer output under `use_logh=True`, where `h = exp(m) > 0` strictly.

Both figures in `EvaluateSolution.ipynb` are affected — the α-sweep mixes 3 conductivity fields with 2 relative-error arrays, and the σ-comparison loads 3 relative-error arrays and then divides by `h_true` a second time.

Per the decision recorded for this plan, the artifacts are **deleted rather than regenerated**: Tier 3 changes the likelihood scale, so any sweep produced now would need re-running with re-tuned α.

**Files:**
- Create: `scripts/make_fixture.py`
- Modify: `.gitignore`
- Delete: `notebooks/hsol_*.npy` (untracked — remove from disk only)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `scripts/make_fixture.py` writing a seeded `test_data/blackbox_output.xdmf`/`.h5`. Tier 3 replaces its noise call with the sensor model but keeps the script's CLI and metadata contract.

- [ ] **Step 1: Write the fixture generator**

Create `scripts/make_fixture.py`:

```python
#!/usr/bin/env python
"""Generate the synthetic 'measured' dataset deterministically.

The observation fixture must be reproducible: an unseeded draw silently changes
the inverse problem's target and makes stored results incomparable.

Usage:
    python scripts/make_fixture.py --seed 0 --sigma 1e-3 --nmesh 128
"""

# type imports
import argparse
import hashlib
import json
from pathlib import Path

# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# local imports
from forward_solver import SteadyHeat2DForwardSolver


def h_true(x):
    """Ground-truth thermal conductivity used to synthesize the observations."""
    return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--nmesh", type=int, default=128)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--dbc", type=float, default=300.0)
    parser.add_argument(
        "--out", type=Path, default=Path("test_data/blackbox_output.xdmf")
    )
    args = parser.parse_args()

    fwd = SteadyHeat2DForwardSolver(
        nmesh=args.nmesh,
        mesh_type="quadrilateral",
        h=h_true,
        q=args.q,
        DBC_value=args.dbc,
    )
    fwd.solve()
    fwd.add_noise(mu=0.0, sigma=args.sigma, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fwd.export_xdmf(str(args.out))

    if MPI.COMM_WORLD.rank == 0:
        digest = hashlib.sha256(
            np.ascontiguousarray(fwd.T_obs.x.array, dtype=np.float64).tobytes()
        ).hexdigest()
        meta = {
            "seed": args.seed,
            "sigma": args.sigma,
            "nmesh": args.nmesh,
            "q": args.q,
            "DBC_value": args.dbc,
            "h_true": "1 + 6*x^2 + x/(1 + 2*y^2)",
            "T_obs_sha256": digest,
        }
        meta_path = args.out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify determinism**

Run it twice and compare the recorded checksum:

```bash
python scripts/make_fixture.py --seed 0 --nmesh 32 --out /tmp/fx1.xdmf
python scripts/make_fixture.py --seed 0 --nmesh 32 --out /tmp/fx2.xdmf
python -c "
import json
a = json.load(open('/tmp/fx1.meta.json'))['T_obs_sha256']
b = json.load(open('/tmp/fx2.meta.json'))['T_obs_sha256']
print('run1', a); print('run2', b); print('deterministic:', a == b)
assert a == b
"
```

Expected: identical digests, `deterministic: True`.

- [ ] **Step 3: Verify a different seed gives different data**

A generator that ignores its seed would also pass Step 2:

```bash
python scripts/make_fixture.py --seed 1 --nmesh 32 --out /tmp/fx3.xdmf
python -c "
import json
a = json.load(open('/tmp/fx1.meta.json'))['T_obs_sha256']
c = json.load(open('/tmp/fx3.meta.json'))['T_obs_sha256']
print('seed 0', a); print('seed 1', c); print('differs:', a != c)
assert a != c
"
```

Expected: `differs: True`.

- [ ] **Step 4: Regenerate the committed fixture**

```bash
python scripts/make_fixture.py --seed 0 --sigma 1e-3 --nmesh 128
git status --short test_data/
```

Expected: `test_data/blackbox_output.h5` modified, `test_data/blackbox_output.meta.json` new. The fixture is now reproducible from the recorded seed.

- [ ] **Step 5: Remove the invalid sweep artifacts**

They are untracked, so this is a disk-only deletion:

```bash
rm -f notebooks/hsol_*.npy
```

Append to `.gitignore`:

```
# Sweep outputs: regenerate with scripts/run_sweep.py (see Tier 3)
notebooks/*.npy
notebooks/*.vtu
```

- [ ] **Step 6: Commit**

```bash
black scripts
git add scripts/make_fixture.py .gitignore test_data/blackbox_output.h5 test_data/blackbox_output.meta.json
git commit -m "Generate the observation fixture deterministically"
```

Note: `test_data/blackbox_output.h5` is committed here deliberately — it is now reproducible from `--seed 0`, which is what makes it reviewable.

---

### Task 6: Make CI enforce the environment and the docs build

CI is red and has been for at least two pushes. `install-and-import.yml` only imports, so it would not have caught a constructor break even with a working VTK; nothing builds the docs, although `.readthedocs.yaml` sets `fail_on_warning: true`, so a docstring slip only surfaces after merge on Read the Docs.

**Files:**
- Modify: `.github/workflows/run-gradchecks.yml`

**Interfaces:**
- Consumes: the pinned environment from Task 1 and the smoke tests from Tasks 2–3.
- Produces: green CI. Later tiers assume CI is a real gate.

- [ ] **Step 1: Add a docs-build job**

In `.github/workflows/run-gradchecks.yml`, append a second job at the same indentation level as the existing `gradcheck` job:

```yaml
  docs:
    name: Build documentation
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install documentation toolchain
        run: pip install -r docs/requirements.txt
      - name: Build docs with warnings as errors
        run: sphinx-build -W --keep-going -b html docs docs/_build/html
```

This job needs no conda: `docs/conf.py` mocks the scientific stack via `autodoc_mock_imports`.

- [ ] **Step 2: Widen the trigger paths**

The workflow currently triggers on `src/**`, `tests/**`, `pyproject.toml`, `environment.yml`. Add the docs and scripts so their changes are also gated. Replace both `paths:` lists:

```yaml
    paths:
      - "src/**"
      - "tests/**"
      - "scripts/**"
      - "docs/**"
      - "pyproject.toml"
      - "environment.yml"
      - ".github/workflows/**"
```

- [ ] **Step 3: Verify the docs job locally**

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Expected: `build succeeded.` with zero warnings.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/run-gradchecks.yml
git commit -m "Build the docs in CI and widen the trigger paths"
git push origin master
```

- [ ] **Step 5: Confirm CI is green — this is the tier's exit gate**

```bash
sleep 60 && gh run list --limit 4
```

Then inspect any failure:

```bash
gh run view <run-id> --log-failed > /tmp/ci.log
python -c "
import re
lines = open('/tmp/ci.log', errors='replace').read().splitlines()
pat = re.compile(r'traceback|error\]|importerror|typeerror|FAILED', re.I)
for l in lines:
    if pat.search(l):
        print(l[-200:])
"
```

Expected: all workflows green. **Do not proceed to Tier 2 until they are** — the whole point of this tier is a trustworthy gate.

---

## Tier 1 Verification Round

Run all of these together before declaring the tier complete.

- [ ] **Environment is pinned and current**

```bash
python -c "import dolfinx; print(dolfinx.__version__)"
```
Expected: `0.11.x`.

- [ ] **Full suite passes**

```bash
pytest -q
```
Expected: `22 passed`.

- [ ] **Gradient checks still hold on the new stack**

```bash
pytest -q -m gradcheck
```
Expected: `3 passed`. These are the correctness gate for the adjoint; they must survive the DOLFINx and PETSc upgrade unchanged.

- [ ] **Compute imports without VTK**

```bash
python -c "
import sys; sys.path.insert(0,'src')
import forward_solver
print('pyvista imported:', 'pyvista' in sys.modules)
"
```
Expected: `pyvista imported: False`.

- [ ] **Fixture is reproducible**

```bash
python -c "
import json
print(json.load(open('test_data/blackbox_output.meta.json')))
"
```
Expected: the metadata records `seed`, `sigma`, `nmesh`, and `T_obs_sha256`.

- [ ] **Docs build clean**

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: `build succeeded.`

- [ ] **CI green**

```bash
gh run list --limit 4
```
Expected: no `failure` on the latest commit.

- [ ] **The invalid artifacts are gone**

```bash
ls notebooks/*.npy 2>/dev/null || echo "no .npy files (correct)"
```
Expected: `no .npy files (correct)`.

**Exit criteria:** every box above checked. Tier 2 assumes a pinned DOLFINx 0.11 environment, green CI, and a lazily-imported plotting stack.
