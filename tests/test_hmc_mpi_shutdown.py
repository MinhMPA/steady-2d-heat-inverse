"""Optional native check: each single-rank process completes MPI finalization."""

# standard library imports
import os
import subprocess
import sys

# testing imports
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("S2DHI_HMC_QUALIFICATION") != "1",
    reason="Native MPI qualification requires S2DHI_HMC_QUALIFICATION=1.",
)

_MPI_SHUTDOWN = """
from mpi4py import MPI

assert MPI.Is_initialized(), 'Normal mpi4py import must initialize MPI.'
provided_thread = MPI.Query_thread()
assert provided_thread == MPI.THREAD_MULTIPLE, (
    f'Posterior MPI qualification requires THREAD_MULTIPLE; got {provided_thread}.'
)
assert MPI.COMM_WORLD.Get_size() == 1, 'Heat solves require one MPI rank.'
MPI.COMM_WORLD.Barrier()
print('S2DHI_BEFORE_MPI_FINALIZE', flush=True)
MPI.Finalize()
assert MPI.Is_finalized()
print('S2DHI_AFTER_MPI_FINALIZE', flush=True)
"""


@pytest.mark.parametrize("attempt", range(3))
def test_single_rank_mpi_process_exits_after_finalization(attempt):
    """Three independent processes must finalize and exit with inherited controls."""
    # A fresh interpreter follows normal mpi4py initialization, independently of
    # pytest's imported MPI state. It inherits transport controls unchanged.
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", _MPI_SHUTDOWN],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"MPI shutdown attempt {attempt} exceeded 30 s; "
            f"stdout={exc.stdout!r}, stderr={exc.stderr!r}"
        )
    evidence = (
        f"MPI shutdown attempt {attempt}: exit={result.returncode}; "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert result.returncode == 0, evidence
    assert "S2DHI_BEFORE_MPI_FINALIZE" in result.stdout, evidence
    assert "S2DHI_AFTER_MPI_FINALIZE" in result.stdout, evidence
