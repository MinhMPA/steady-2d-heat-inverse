# Installation

## Requirements

Python ≥ 3.11. The scientific stack (DOLFINx, PETSc, MPI) is **conda-only** — there is no
working pip-only installation path. All runtime dependencies are declared in
`environment.yml`; `pyproject.toml` deliberately declares no runtime `dependencies`.

| Layer | Packages |
|---|---|
| FEM | `fenics-dolfinx`, `fenics-ufl`, `fenics-basix` |
| Linear algebra / optimizer | `petsc4py` (KSP + TAO) |
| Parallelism | `mpi4py` |
| Interpolation / arrays | `numpy`, `scipy`, `pandas` |
| Visualization / IO | `pyvista<=0.45.3`, `h5py` |
| Notebooks | `jupyterlab` |

## From `environment.yml`

```bash
conda env create -n steady-2d-heat-inverse -f environment.yml
conda activate steady-2d-heat-inverse
pip install -e ".[dev]"
```

`mamba` or `micromamba` can be substituted for `conda`. The `[dev]` extra adds `pytest`
and `pytest-xdist`, needed to run the gradient tests.

:::{note}
`environment.yml` declares its own name (`steady-2d-heat`); the `-n` flag above overrides
it so the environment matches the name used by the CI workflows.
:::

## The flat-module import convention

`pyproject.toml` installs the contents of `src/` as **top-level modules** rather than as a
package:

```toml
[tool.setuptools]
package-dir = {"" = "src"}
py-modules = [
  "forward_solver", "adjoint_solver", "tao_solver",
  "domain_coefficient", "plotting_utils",
]
```

So imports are flat:

```python
from forward_solver import SteadyHeat2DForwardSolver   # correct
from src.forward_solver import SteadyHeat2DForwardSolver  # wrong
```

`src/__init__.py` exists but is vestigial — its relative imports are never exercised under
this layout.

## Verifying the install

```bash
python -c "import forward_solver, adjoint_solver, tao_solver; print('OK')"
```

This is exactly the smoke test run by the `install-and-import` GitHub Actions workflow on
every push.

## Building the documentation

The docs build needs none of the scientific stack — it is mocked via
`autodoc_mock_imports` in `docs/conf.py`:

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```
