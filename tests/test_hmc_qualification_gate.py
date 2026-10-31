"""Optional sampling qualification must preserve the core heat-validation suite."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

CONNECTION_TESTS = Path(__file__).with_name("test_hmc_jax_connection.py")
QUALIFICATION_FLAG = "S2DHI_HMC_QUALIFICATION"


def run_gate_child(code, *, qualification=False):
    env = os.environ.copy()
    # A qualified parent suite must still exercise the ordinary core behavior.
    env.pop(QUALIFICATION_FLAG, None)
    env.pop("JAX_ENABLE_X64", None)
    env.pop("JAX_PLATFORMS", None)
    if qualification:
        env[QUALIFICATION_FLAG] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(CONNECTION_TESTS)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_collection_skips_sampling_before_importing_optional_dependencies():
    run_gate_child("""
        import importlib.abc
        import runpy
        import sys
        import pytest
        attempted = []
        class MissingSampling(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in {"jax", "jaxlib", "blackjax"}:
                    attempted.append(fullname)
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, MissingSampling())
        with pytest.raises(pytest.skip.Exception) as exc:
            runpy.run_path(sys.argv[1])
        assert "S2DHI_HMC_QUALIFICATION=1" in str(exc.value)
        assert not attempted
        assert "hmc_blackjax" not in sys.modules
    """)


def test_requested_qualification_cannot_skip_missing_sampling_dependencies():
    run_gate_child(
        """
        import importlib.abc
        import runpy
        import sys
        import pytest
        class MissingSampling(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in {"jax", "jaxlib", "blackjax"}:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, MissingSampling())
        with pytest.raises(ImportError, match="sampling"):
            runpy.run_path(sys.argv[1])
        """,
        qualification=True,
    )


@pytest.mark.parametrize(
    "message",
    [
        "Sampling qualification requires jax==0.11.2; found an unqualified version.",
        "Sampling requires real float64 JAX arrays.",
        "Sampling requires the eager CPU backend.",
    ],
)
def test_requested_qualification_surfaces_environment_failure_during_collection(
    message,
):
    # Keep this collection-boundary check usable in the JAX-free core environment.
    # The actual float64 probability and force are checked by the qualified module.
    run_gate_child(
        f"""
        import runpy
        import sys
        import pytest
        import hmc_blackjax
        failure = RuntimeError({message!r})
        def reject_environment():
            raise failure
        hmc_blackjax.require_sampling_environment = reject_environment
        with pytest.raises(RuntimeError) as exc:
            runpy.run_path(sys.argv[1])
        assert exc.value is failure
        """,
        qualification=True,
    )
