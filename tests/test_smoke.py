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
