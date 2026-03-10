from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem

from adjoint_solver import SteadyHeat2DAdjointSolver
from forward_solver import SteadyHeat2DForwardSolver
from tao_solver import SteadyHeat2DTAOSolver

CoefficientLike = Any

DEFAULT_PETSC_OPTIONS = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    h_true: CoefficientLike
    h0: CoefficientLike
    q: CoefficientLike = 1.0
    sigma: float = 1.0
    alpha: float = 1e-4
    nmesh: int = 16
    mesh_type: str = "quadrilateral"
    T_bottom: float = 300.0
    observation_noise: float = 0.0
    observation_seed: int | None = None
    truth_petsc_opts: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkMethod:
    name: str
    forward_petsc_opts: dict[str, Any] | None = None
    adjoint_petsc_opts: dict[str, Any] | None = None
    adaptive_ksp_rtol: dict[str, Any] | None = None
    tao_type: str = "blmvm"
    ls_algorithm: str = "armijo"
    use_logh: bool = True
    h_min: float | None = 0.1
    h_max: float | None = 10.0
    gatol: float = 1e-6
    grtol: float = 1e-4
    gttol: float = 1e-4
    mit: int = 50
    options: bool = False
    verbose: int = 0


@dataclass(slots=True)
class BenchmarkRunResult:
    case: BenchmarkCase
    method: BenchmarkMethod
    summary: dict[str, Any]
    trace: list[dict[str, Any]]
    solution: np.ndarray
    truth: np.ndarray

    @property
    def case_name(self) -> str:
        return self.case.name

    @property
    def method_name(self) -> str:
        return self.method.name


def run_benchmark_case(
    case: BenchmarkCase,
    method: BenchmarkMethod,
) -> BenchmarkRunResult:
    fwd_truth, fwd, T_obs, adj = build_inverse_problem(case, method)

    trace: list[dict[str, Any]] = []
    start_time = perf_counter()

    def monitor(tao):
        its, obj, gnorm, cnorm, xdiff, reason = tao.getSolutionStatus()
        trace.append(
            {
                "case_name": case.name,
                "method_name": method.name,
                "tao_iter": int(its),
                "wall_time_s": perf_counter() - start_time,
                "objective": float(obj),
                "gradient_norm": float(gnorm),
                "constraint_norm": float(cnorm),
                "step_length": float(xdiff),
                "converged_reason": int(reason),
                "forward_ksp_rtol": float(fwd.get_ksp_rtol()),
                "adjoint_ksp_rtol": float(adj.get_ksp_rtol()),
                "forward_ksp_iterations": int(fwd.problem.solver.getIterationNumber()),
                "adjoint_ksp_iterations": int(adj.problem.solver.getIterationNumber()),
            }
        )

    tao = SteadyHeat2DTAOSolver(
        fwd,
        adj,
        tao_type=method.tao_type,
        ls_algorithm=method.ls_algorithm,
        use_logh=method.use_logh,
        h_min=method.h_min,
        h_max=method.h_max,
        gatol=method.gatol,
        grtol=method.grtol,
        gttol=method.gttol,
        mit=method.mit,
        monitor=monitor,
        adaptive_ksp_rtol=method.adaptive_ksp_rtol,
        options=method.options,
        verbose=method.verbose,
    )
    solution = np.array(tao.solve(), copy=True)
    wall_time_s = perf_counter() - start_time

    truth = _coefficient_array(fwd_truth.h.function, fwd.V)
    h_error_l2_rel = _relative_l2_error(solution, truth)
    status = tao.tao.getSolutionStatus()
    total_forward_ksp_iterations = sum(row["forward_ksp_iterations"] for row in trace)
    total_adjoint_ksp_iterations = sum(row["adjoint_ksp_iterations"] for row in trace)
    summary = {
        "case_name": case.name,
        "method_name": method.name,
        "wall_time_s": wall_time_s,
        "tao_iterations": int(status[0]),
        "final_objective": float(status[1]),
        "final_gradient_norm": float(status[2]),
        "final_constraint_norm": float(status[3]),
        "final_step_length": float(status[4]),
        "converged_reason": int(status[5]),
        "final_forward_ksp_rtol": float(fwd.get_ksp_rtol()),
        "final_adjoint_ksp_rtol": float(adj.get_ksp_rtol()),
        "final_ksp_rtol": float(fwd.get_ksp_rtol()),
        "total_forward_ksp_iterations": int(total_forward_ksp_iterations),
        "total_adjoint_ksp_iterations": int(total_adjoint_ksp_iterations),
        "total_ksp_iterations": int(
            total_forward_ksp_iterations + total_adjoint_ksp_iterations
        ),
        "h_error_l2_rel": h_error_l2_rel,
        "ksp_tightenings": 0
        if tao.ksp_rtol_controller is None
        else int(tao.ksp_rtol_controller.num_tightenings),
        "ksp_rtol_history": [float(fwd.get_ksp_rtol())]
        if tao.ksp_rtol_controller is None
        else [float(val) for val in tao.ksp_rtol_controller.history],
        "initial_objective": float(trace[0]["objective"]) if trace else float(status[1]),
    }
    return BenchmarkRunResult(
        case=case,
        method=method,
        summary=summary,
        trace=trace,
        solution=solution,
        truth=truth,
    )


def run_benchmark_suite(
    cases: list[BenchmarkCase],
    methods: list[BenchmarkMethod],
) -> list[BenchmarkRunResult]:
    return [run_benchmark_case(case, method) for case in cases for method in methods]


def benchmark_summaries(results: list[BenchmarkRunResult]) -> list[dict[str, Any]]:
    return [dict(result.summary) for result in results]


def benchmark_trace_rows(results: list[BenchmarkRunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(dict(row) for row in result.trace)
    return rows


def build_inverse_problem(
    case: BenchmarkCase,
    method: BenchmarkMethod,
) -> tuple[
    SteadyHeat2DForwardSolver,
    SteadyHeat2DForwardSolver,
    fem.Function,
    SteadyHeat2DAdjointSolver,
]:
    truth_petsc_opts = DEFAULT_PETSC_OPTIONS | (case.truth_petsc_opts or {})
    fwd_truth = SteadyHeat2DForwardSolver(
        nmesh=case.nmesh,
        mesh_type=case.mesh_type,
        h=case.h_true,
        q=case.q,
        DBC_value=case.T_bottom,
        petsc_opts=truth_petsc_opts,
    )
    fwd_truth.solve()

    forward_petsc_opts = DEFAULT_PETSC_OPTIONS | (method.forward_petsc_opts or {})
    fwd = SteadyHeat2DForwardSolver(
        nmesh=case.nmesh,
        mesh_type=case.mesh_type,
        h=case.h0,
        q=case.q,
        DBC_value=case.T_bottom,
        petsc_opts=forward_petsc_opts,
    )
    fwd.solve()

    if case.observation_noise > 0.0:
        T_obs = fwd_truth.add_noise(
            mu=0.0, sigma=case.observation_noise, seed=case.observation_seed
        )
    else:
        T_obs = fem.Function(fwd.V, name="ObservedTemperature")
        T_obs.x.array[:] = fwd_truth.T.x.array
        T_obs.x.scatter_forward()

    adj = SteadyHeat2DAdjointSolver(
        fwd,
        T_obs,
        sigma=case.sigma,
        alpha=case.alpha,
        DBC_value=0.0,
        petsc_opts=method.adjoint_petsc_opts,
    )
    return fwd_truth, fwd, T_obs, adj


def evaluate_objective(
    forward: SteadyHeat2DForwardSolver,
    T_obs: fem.Function,
    sigma: float,
    alpha: float,
) -> float:
    dx = ufl.Measure("dx", domain=forward.mesh)
    misfit = (
        0.5
        * (1.0 / sigma**2)
        * ufl.inner(forward.T - T_obs, forward.T - T_obs)
        * dx
    )
    regularization = (
        0.5
        * alpha
        * ufl.inner(ufl.grad(forward.h.function), ufl.grad(forward.h.function))
        * dx
    )
    return fem.assemble_scalar(fem.form(misfit + regularization))


def default_benchmark_cases(
    *,
    nmesh: int = 24,
    sigma: float = 5e-3,
    alpha: float = 1e-4,
    observation_noise: float | None = None,
) -> list[BenchmarkCase]:
    obs_noise = sigma if observation_noise is None else observation_noise
    return [
        BenchmarkCase(
            name="smooth-low-contrast",
            h_true=_smooth_low_contrast_h,
            h0=_default_initial_guess_h,
            nmesh=nmesh,
            sigma=sigma,
            alpha=alpha,
            observation_noise=obs_noise,
            observation_seed=17,
        ),
        BenchmarkCase(
            name="smooth-high-contrast",
            h_true=_smooth_high_contrast_h,
            h0=_default_initial_guess_h,
            nmesh=nmesh,
            sigma=sigma,
            alpha=alpha,
            observation_noise=obs_noise,
            observation_seed=23,
        ),
        BenchmarkCase(
            name="sharp-interface",
            h_true=_sharp_interface_h,
            h0=_default_initial_guess_h,
            nmesh=nmesh,
            sigma=sigma,
            alpha=alpha,
            observation_noise=obs_noise,
            observation_seed=31,
        ),
    ]


def default_benchmark_methods(*, mit: int = 40) -> list[BenchmarkMethod]:
    return [
        BenchmarkMethod(
            name="fixed-rtol-1e-04",
            forward_petsc_opts={"ksp_rtol": 1e-4},
            mit=mit,
        ),
        BenchmarkMethod(
            name="fixed-rtol-1e-06",
            forward_petsc_opts={"ksp_rtol": 1e-6},
            mit=mit,
        ),
        BenchmarkMethod(
            name="fixed-rtol-1e-08",
            forward_petsc_opts={"ksp_rtol": 1e-8},
            mit=mit,
        ),
        BenchmarkMethod(
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
            mit=mit,
        ),
        BenchmarkMethod(
            name="adaptive-rtol-1e-06",
            adaptive_ksp_rtol={
                "initial_rtol": 1e-6,
                "min_rtol": 1e-8,
                "tighten_factor": 0.1,
                "stall_iterations": 2,
                "stall_rel_improve": 0.05,
                "rejection_iterations": 2,
                "rejected_step_threshold": 0.25,
            },
            mit=mit,
        ),
    ]


def _coefficient_array(coefficient: Any, V: fem.FunctionSpace) -> np.ndarray:
    if isinstance(coefficient, fem.Function):
        return np.array(coefficient.x.array, copy=True)
    if isinstance(coefficient, fem.Constant):
        value = coefficient.value
    else:
        value = coefficient
    f = fem.Function(V)
    f.interpolate(
        lambda x: np.full(x.shape[1], value, dtype=default_scalar_type)
    )
    return np.array(f.x.array, copy=True)


def _relative_l2_error(sol: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(sol - truth) / np.linalg.norm(truth))


def _default_initial_guess_h(x):
    return 1.6 + 0.45 * x[0] + 0.25 * x[1]


def _smooth_low_contrast_h(x):
    return 1.0 + 0.35 * x[0] + 0.25 * x[1] + 0.15 * x[0] * x[1]


def _smooth_high_contrast_h(x):
    return 0.9 + 2.8 * ufl.exp(-((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2) / 0.025)


def _sharp_interface_h(x):
    return ufl.conditional(ufl.lt(x[0] + 0.25 * x[1], 0.62), 1.0, 3.0)
