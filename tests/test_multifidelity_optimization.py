import sys
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_utils import BenchmarkCase, BenchmarkMethod, run_benchmark_case
from ._helpers import h0, h_true


N_MESH = 8
T_BOTTOM = 300.0
NOISE_SIGMA = 1.0
REG_ALPHA = 1e-4


def _small_case() -> BenchmarkCase:
    return BenchmarkCase(
        name="small-smooth",
        h_true=h_true,
        h0=h0,
        nmesh=N_MESH,
        sigma=NOISE_SIGMA,
        alpha=REG_ALPHA,
        observation_noise=0.0,
        T_bottom=T_BOTTOM,
    )


def test_adaptive_multifidelity_trace_keeps_forward_and_adjoint_synced():
    case = _small_case()
    method = BenchmarkMethod(
        name="adaptive-aggressive",
        adaptive_ksp_rtol={
            "initial_rtol": 1e-4,
            "min_rtol": 1e-8,
            "tighten_factor": 0.1,
            "stall_iterations": 1,
            "stall_rel_improve": 0.999,
            "rejection_iterations": 99,
            "rejected_step_threshold": 0.0,
        },
        mit=6,
    )

    result = run_benchmark_case(case, method)
    trace = result.trace
    history = result.summary["ksp_rtol_history"]

    assert len(trace) >= 2
    assert len(history) >= 2
    assert result.summary["ksp_tightenings"] >= 1
    assert np.isclose(history[0], 1e-4)
    assert np.isclose(
        result.summary["final_forward_ksp_rtol"],
        result.summary["final_adjoint_ksp_rtol"],
    )
    assert np.all(
        [
            np.isclose(row["forward_ksp_rtol"], row["adjoint_ksp_rtol"])
            for row in trace
        ]
    )
    forward_rtols = [row["forward_ksp_rtol"] for row in trace]
    assert np.all(np.diff(forward_rtols) <= 1e-15)
    assert result.summary["total_forward_ksp_iterations"] > 0
    assert result.summary["total_adjoint_ksp_iterations"] > 0


def test_adaptive_multifidelity_reduces_ksp_work_vs_fixed_baseline():
    case = _small_case()
    fixed_method = BenchmarkMethod(
        name="fixed-rtol-1e-10",
        forward_petsc_opts={"ksp_rtol": 1e-10},
        mit=12,
    )
    adaptive_method = BenchmarkMethod(
        name="adaptive-rtol-1e-04",
        adaptive_ksp_rtol={
            "initial_rtol": 1e-4,
            "min_rtol": 1e-8,
            "tighten_factor": 0.1,
            "stall_iterations": 2,
            "stall_rel_improve": 0.05,
            "rejection_iterations": 2,
            "rejected_step_threshold": 0.25,
        },
        mit=12,
    )

    fixed = run_benchmark_case(case, fixed_method)
    adaptive = run_benchmark_case(case, adaptive_method)

    assert adaptive.summary["total_ksp_iterations"] < fixed.summary["total_ksp_iterations"]
    assert adaptive.summary["final_objective"] <= 1.10 * fixed.summary["final_objective"]
    assert adaptive.summary["h_error_l2_rel"] <= 1.10 * fixed.summary["h_error_l2_rel"]
    assert adaptive.summary["ksp_tightenings"] >= 1
