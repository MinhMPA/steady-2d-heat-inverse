# Installation

For conductivity sampling, use the source tree containing this guide and
`src/heat_inference.py` (the `docs/hmc-inference-guide` branch). The reviewed
sampling source is `3df71cacf650f94d77c54fc0dbcf0f916825e199`. No public
branch download or `master` installation is assumed here; the commands below
run from an already available checkout. See the [dated evidence](validation/hmc-status.md)
for the distinction between local verification and publication.

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

Preserve the original reconstruction environment. The measured sampling
qualification uses CPython 3.12, eager CPU float64 and one MPI rank, with BlackJAX 1.7.1, JAX/JAXlib 0.11.2
and ArviZ Stats 1.3.3. The base project accepts Python 3.11, but that is not the
qualified sampling environment. Start from `environment.yml` in a separate environment, or clone the verified original
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
Install only the optional layers needed for the calculation:

| Extra | Purpose | Native heat solve needed? |
| --- | --- | --- |
| `[sampling]` | BlackJAX/JAX trajectory machinery | Yes, for the physical posterior |
| `[diagnostics]` | ArviZ Stats convergence and Monte Carlo errors | No, for saved histories |
| `[validation]` | Matplotlib physical-field figures | No, for exported arrays |
| `[plot]` | PyVista/VTK finite-element figures | Requires separate native-library qualification |

Reading saved conductivity arrays and summaries needs none of these extras.
These optional dependencies are lazy: legacy imports and reconstruction work
without sampling, diagnostics or plotting installed.

Before starting Python or a notebook kernel, configure precision and threads as
below. For the dedicated macOS ARM64 MPICH environment, first set
`export FI_PROVIDER=tcp` in the same shell, following the
[transport and shutdown instructions](#macos-mpich-transport-and-shutdown).

```bash
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export S2DHI_HMC_QUALIFICATION=1
export PYTHONPATH="$PWD/src"
python -c "from hmc_blackjax import require_sampling_environment; print(require_sampling_environment())"
```

Use one MPI rank per heat solve; do not launch the posterior notebook with
`mpirun -n 4`. Four independent chains are requested with `chains=4`.

### macOS MPICH transport and shutdown

For the dedicated macOS ARM64 posterior environment with MPICH 5.0.1
(`ch4:ofi`, libfabric 2.6.0), select its TCP transport **before starting Python,
Jupyter, or any finite-element calculation**:

```bash
conda activate s2dhi-hmc
export FI_PROVIDER=tcp
export S2DHI_HMC_QUALIFICATION=1
export PYTHONPATH="$PWD/src"
python - <<'PY'
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_hmc_mpi_shutdown.py"],
    check=True, timeout=150,
)
PY
```

To retain the provider choice for future activations of this environment:

```bash
conda env config vars set -n s2dhi-hmc FI_PROVIDER=tcp
```

Reactivate the environment before starting a new Python or Jupyter process.
An already initialized MPI process cannot adopt this startup change.

The optional check requires three independent single-rank processes to use
normal mpi4py initialization with `THREAD_MULTIPLE`, pass a barrier, return from
explicit `MPI.Finalize()`, print a subsequent marker and exit
successfully. Each process has a 30-second limit; the command above also bounds
the entire check to 150 seconds. A timeout fails qualification. A saved posterior
trajectory alone does not establish that its process shut down cleanly.

The setting selects an existing libfabric provider; it is an explicit environment
configuration, rather than a repair to MPICH or libfabric source. Local minimal
MPI checks found stalled finalization with the default sockets transport on
several VPN interfaces, with successful exits using TCP on the paired interfaces.
This supports the transport choice without reproducing the identical historical
shutdown error. The related upstream report is
[MPICH issue 7760](https://github.com/pmodels/mpich/issues/7760).
The [measured qualification record](validation/hmc-mpi-shutdown.md) gives the
paired experiments, physical comparisons and limits.

Set this only for the dedicated posterior environment. Other MPI installations,
including Linux, need their own qualification. Start with the provider's normal
interface choice: forcing `lo0` failed in the local checks. Declare
`FI_TCP_IFACE` or `FI_SOCKETS_IFACE` only after qualifying that particular choice.
Fresh child processes inherit these controls. The library leaves them under the
caller's control and uses normal MPI finalization, with errors visible.

The saved numerical record includes `FI_PROVIDER`, `FI_TCP_IFACE` and
`FI_SOCKETS_IFACE`, including whether each is unset. Changing any declaration
prevents continuation in the same archive. Existing archives remain readable
without MPI or the original environment; a different numerical identity requires
a new calculation directory. Keep the original reconstruction environment and
its MPI installation intact.

### Measured macOS compiler setup

The dedicated macOS environment inherited compiler flags from an unrelated conda
calculation. Those flags produced duplicate-RPATH libraries during finite-element
form compilation, which can surface as a duplicate `LC_RPATH` native-library
load error. Cached compiled forms can conceal this problem. The measured local
checks cleared them and used Apple's native compiler with an explicit bundle linker command:

```bash
unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS CPATH LIBRARY_PATH
export CC=/usr/bin/clang CXX=/usr/bin/clang++
export LDSHARED='/usr/bin/clang -bundle -undefined dynamic_lookup'
export XDG_CACHE_HOME="$(mktemp -d /tmp/hmc-compile.XXXXXX)"
python - <<'PY'
import subprocess, sys
subprocess.run(
    [sys.executable, "-c",
     "from forward_solver import SteadyHeat2DForwardSolver; "
     "f=SteadyHeat2DForwardSolver(nmesh=2,h=1.0,q=1.0); "
     "f.solve(); print('fresh-cache heat solve passed')"],
    check=True, timeout=120,
)
PY
```

This fresh-cache probe checks real form compilation rather than reusing an old
compiled form. Record the actual compiler, architecture, numerical packages,
source revision and thread settings with validation outcomes. Linux CI uses its
own compiler and linker; the macOS linker command above is platform-specific.
PyVista/VTK is deliberately absent from the FEM environment because of the known
native-library conflict. Use the notebook's lightweight Matplotlib renderer or
exported NumPy fields in a separate plotting environment.

A fresh-cache compilation must complete **and its process must exit** within
the declared bound. It complements the three independent MPI-exit checks; it
does not establish posterior convergence. Effective local/user DOLFINx and FFCx
settings now enter continuation identity, correcting the omission in the
[16 November review](validation/hmc-status.md#numerical-reproducibility).
Changing compiler settings requires constructing a new inverse problem and a
new calculation; older archives remain available for read-only analysis.
