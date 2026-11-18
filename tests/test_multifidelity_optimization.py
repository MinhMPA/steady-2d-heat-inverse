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
        [np.isclose(row["forward_ksp_rtol"], row["adjoint_ksp_rtol"]) for row in trace]
    )
    forward_rtols = [row["forward_ksp_rtol"] for row in trace]
    assert np.all(np.diff(forward_rtols) <= 1e-15)
    assert result.summary["total_forward_ksp_iterations"] > 0
    assert result.summary["total_adjoint_ksp_iterations"] > 0


def test_adaptive_multifidelity_preserves_reconstruction_accuracy():
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

    # Cheaper individual heat solves need not reduce total work: a looser
    # gradient can require more line-search trials. Report, rather than assume,
    # the measured cost of both reconstructions.
    assert adaptive.summary["total_ksp_iterations"] > 0
    assert fixed.summary["total_ksp_iterations"] > 0
    assert (
        adaptive.summary["final_objective"] <= 1.10 * fixed.summary["final_objective"]
    )
    assert adaptive.summary["h_error_l2_rel"] <= 1.10 * fixed.summary["h_error_l2_rel"]
    assert adaptive.summary["ksp_tightenings"] >= 1


def test_noisy_observations_build_on_the_forward_solvers_space():
    """`observation_noise > 0` must produce a usable inverse problem.

    `add_noise` returns a Function on the *truth* solver's space, but the adjoint
    solver asserts its observation lives on the space of the forward solver being
    optimized. Copying the DOF values across is what keeps the two consistent;
    without it this path raises AssertionError before any optimization starts.
    """
    case = BenchmarkCase(
        name="small-noisy",
        h_true=h_true,
        h0=h0,
        nmesh=N_MESH,
        sigma=NOISE_SIGMA,
        alpha=REG_ALPHA,
        observation_noise=1e-3,
        observation_seed=0,
        T_bottom=T_BOTTOM,
    )
    method = BenchmarkMethod(
        name="fixed", forward_petsc_opts={"ksp_rtol": 1e-10}, mit=3
    )

    result = run_benchmark_case(case, method)

    assert np.isfinite(result.summary["final_objective"])
    assert np.isfinite(result.solution).all()
    assert result.solution.shape == result.truth.shape


def test_benchmark_counts_every_line_search_heat_and_adjoint_solve(monkeypatch):
    import benchmark_utils as benchmark

    original = benchmark.build_inverse_problem
    measured = {"forward": [], "adjoint": []}

    def instrument(*args):
        truth, forward, observations, adjoint = original(*args)
        for name, equation in (("forward", forward), ("adjoint", adjoint)):
            solve = equation.problem.solve

            def counted(solve=solve, equation=equation, name=name):
                result = solve()
                measured[name].append(equation.problem.solver.getIterationNumber())
                return result

            monkeypatch.setattr(equation.problem, "solve", counted)
        return truth, forward, observations, adjoint

    monkeypatch.setattr(benchmark, "build_inverse_problem", instrument)
    result = benchmark.run_benchmark_case(_small_case(), BenchmarkMethod(name="line-search", mit=50))
    for name, work in measured.items():
        assert result.summary[f"total_{name}_ksp_iterations"] == sum(work)
        assert result.summary[f"total_{name}_solves"] == len(work)
    # This case actually evaluates trial points outside the monitor trace.
    assert len(measured["forward"]) > len(result.trace)


def test_failed_assembly_does_not_recount_previous_ksp_iterations(monkeypatch):
    from benchmark_utils import build_inverse_problem
    from ._hmc_helpers import _release_problem

    truth, forward, observations, adjoint = build_inverse_problem(
        _small_case(), BenchmarkMethod(name="assembly-failure")
    )
    try:
        for equation in (forward, adjoint):
            equation.solve()
            previous = equation.total_ksp_iterations
            attempts = equation.total_linear_solves
            assert previous > 0

            def fail():
                raise RuntimeError("assembly failed before KSP")

            monkeypatch.setattr(equation.problem, "solve", fail)
            import pytest
            with pytest.raises(RuntimeError, match="assembly failed"):
                equation.solve()
            assert equation.total_ksp_iterations == previous
            assert equation.total_linear_solves == attempts + 1
    finally:
        for equation in (truth, forward, adjoint):
            _release_problem(equation.problem)
