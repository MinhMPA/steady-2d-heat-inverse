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
| IO | `h5py` |
| Optional visualization | Matplotlib `[validation]`; PyVista `[plot]` in a separate environment |
| Notebooks | `jupyterlab` |

## From `environment.yml`

```bash
conda env create -n steady-2d-heat-inverse -f environment.yml
conda activate steady-2d-heat-inverse
pip install -e ".[dev]"
```

`mamba` or `micromamba` can be substituted for `conda`. The `[dev]` extra adds `pytest`
and `pytest-xdist`, plus `nbformat`, `nbclient` and `ipykernel` for the executable
conductivity example. Matplotlib belongs only to the optional `[validation]` extra.

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

## Separate CPU posterior environment

Preserve the original reconstruction environment. Sampling is qualified for Python
3.12+, eager CPU float64, with pinned JAX/BlackJAX and optional ArviZ Stats. Start
from `environment.yml` in a separate environment, or clone the verified original
`s2dhi-011` environment as `s2dhi-hmc`. Resolve with `constraints/hmc-cpu-py312.txt` and inspect a dry-run report
before installation:

```bash
conda create -n s2dhi-hmc --clone s2dhi-011
conda activate s2dhi-hmc
python -m pip install --dry-run --report /tmp/hmc-resolution.json \
  -c constraints/hmc-cpu-py312.txt -e ".[dev,sampling,diagnostics,validation]"
# Inspect the report: do not replace the conda FEM/numerical stack through pip.
python -m pip install -c constraints/hmc-cpu-py312.txt \
  -e ".[dev,sampling,diagnostics,validation]"
```

The constrained numerical versions are NumPy 2.5.2, SciPy 1.18.0, petsc4py 3.25.4,
DOLFINx 0.11.0 and h5py 3.16.0. If a fresh conda resolution differs, construct a
compatible native conda environment or requalify it; do not use pip to replace its
FEM libraries. Sampling pins are BlackJAX 1.7.1, JAX/JAXlib 0.11.2; diagnostics use
`arviz-stats==1.3.3`. `[validation]` adds Matplotlib for portable physical figures.
These optional dependencies are lazy: legacy imports and reconstruction work
without sampling, diagnostics or plotting installed.

Before starting Python or a notebook kernel:

```bash
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export S2DHI_HMC_QUALIFICATION=1
export PYTHONPATH="$PWD/src"
python -c "from hmc_blackjax import require_sampling_environment; print(require_sampling_environment())"
```

Use one MPI rank per heat solve; do not launch the posterior notebook with
`mpirun -n 4`. Four independent chains are requested with `chains=4`.

### Measured macOS compiler setup

The dedicated macOS environment inherited compiler flags from an unrelated conda
calculation. Those flags produced duplicate-RPATH libraries during finite-element
form compilation. The measured local checks cleared them and used Apple's native
compiler with an explicit bundle linker command:

```bash
unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS CPATH LIBRARY_PATH
export CC=/usr/bin/clang CXX=/usr/bin/clang++
export LDSHARED='/usr/bin/clang -bundle -undefined dynamic_lookup'
export XDG_CACHE_HOME="$(mktemp -d /tmp/hmc-compile.XXXXXX)"
python -c "from forward_solver import SteadyHeat2DForwardSolver; f=SteadyHeat2DForwardSolver(nmesh=2,h=1.0,q=1.0); f.solve(); print('fresh-cache heat solve passed')"
```

This fresh-cache probe checks real form compilation rather than reusing an old
compiled form. Record the actual compiler, architecture, numerical packages,
source revision and thread settings with validation outcomes. Linux CI uses its
own compiler and linker; the macOS linker command above is platform-specific.
PyVista/VTK is deliberately absent from the FEM environment because of the known
native-library conflict. Use the notebook's lightweight Matplotlib renderer or
exported NumPy fields in a separate plotting environment.
