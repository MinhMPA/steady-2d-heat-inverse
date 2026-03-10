from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Protocol


class _KSPToleranceSolver(Protocol):
    def get_ksp_rtol(self) -> float: ...

    def set_ksp_rtol(self, rtol: float) -> None: ...


@dataclass(frozen=True, slots=True)
class AdaptiveKSPRTOLSettings:
    """
    Settings for a shared adaptive KSP relative tolerance schedule.
    """

    initial_rtol: float | None = None
    min_rtol: float = 1e-10
    tighten_factor: float = 0.1
    stall_iterations: int = 3
    stall_rel_improve: float = 5e-2
    rejection_iterations: int = 2
    rejected_step_threshold: float = 2.5e-1

    def __post_init__(self) -> None:
        if self.initial_rtol is not None and self.initial_rtol <= 0.0:
            raise ValueError("initial_rtol must be positive.")
        if self.min_rtol <= 0.0:
            raise ValueError("min_rtol must be positive.")
        if not (0.0 < self.tighten_factor < 1.0):
            raise ValueError("tighten_factor must lie in (0, 1).")
        if self.stall_iterations < 1:
            raise ValueError("stall_iterations must be >= 1.")
        if self.rejection_iterations < 1:
            raise ValueError("rejection_iterations must be >= 1.")
        if not (0.0 <= self.stall_rel_improve < 1.0):
            raise ValueError("stall_rel_improve must lie in [0, 1).")
        if self.rejected_step_threshold < 0.0:
            raise ValueError("rejected_step_threshold must be non-negative.")


class SharedKSPToleranceController:
    """
    Keep a shared KSP relative tolerance synchronized across multiple solvers and
    tighten it when optimization progress stalls.
    """

    def __init__(
        self,
        solvers: Iterable[_KSPToleranceSolver],
        settings: AdaptiveKSPRTOLSettings | None = None,
    ):
        self.solvers = tuple(solvers)
        if not self.solvers:
            raise ValueError("At least one solver is required.")

        self.settings = settings or AdaptiveKSPRTOLSettings()
        initial_rtol = (
            self.settings.initial_rtol
            if self.settings.initial_rtol is not None
            else self.solvers[0].get_ksp_rtol()
        )
        if initial_rtol < self.settings.min_rtol:
            raise ValueError("initial_rtol must be >= min_rtol.")

        self.current_rtol = float(initial_rtol)
        self.history = [self.current_rtol]
        self.num_tightenings = 0

        self._prev_gradient_norm: float | None = None
        self._stall_count = 0
        self._rejection_count = 0

        self._apply(self.current_rtol)

    def _apply(self, rtol: float) -> None:
        for solver in self.solvers:
            solver.set_ksp_rtol(rtol)

    def observe(
        self,
        *,
        iteration: int,
        gradient_norm: float | None,
        step_length: float | None = None,
    ) -> bool:
        """
        Observe one optimization iteration and tighten the shared KSP tolerance if
        the reported gradient norm stalls or the accepted step length repeatedly
        drops below the configured threshold.
        """
        if iteration <= 0:
            self._prev_gradient_norm = _sanitize_metric(gradient_norm)
            return False

        gradient_norm = _sanitize_metric(gradient_norm)
        if gradient_norm is not None and self._prev_gradient_norm is not None:
            denom = max(abs(self._prev_gradient_norm), 1e-32)
            rel_improve = max(self._prev_gradient_norm - gradient_norm, 0.0) / denom
            if rel_improve < self.settings.stall_rel_improve:
                self._stall_count += 1
            else:
                self._stall_count = 0
        elif gradient_norm is None:
            self._stall_count = 0

        if step_length is not None and step_length <= self.settings.rejected_step_threshold:
            self._rejection_count += 1
        else:
            self._rejection_count = 0

        tightened = False
        if self.current_rtol > self.settings.min_rtol and (
            self._stall_count >= self.settings.stall_iterations
            or self._rejection_count >= self.settings.rejection_iterations
        ):
            new_rtol = max(
                self.settings.min_rtol, self.current_rtol * self.settings.tighten_factor
            )
            if new_rtol < self.current_rtol:
                self.current_rtol = float(new_rtol)
                self.history.append(self.current_rtol)
                self.num_tightenings += 1
                self._apply(self.current_rtol)
                tightened = True
            self._stall_count = 0
            self._rejection_count = 0

        self._prev_gradient_norm = gradient_norm
        return tightened

    def observe_tao_status(
        self,
        status: tuple[int, float, float, float, float, int],
    ) -> bool:
        """
        Consume PETSc.TAO.getSolutionStatus() output.
        """
        its, _f, gnorm, _cnorm, xdiff, _reason = status
        return self.observe(iteration=its, gradient_norm=gnorm, step_length=xdiff)


def _sanitize_metric(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not isfinite(value):
        return None
    return abs(value)
