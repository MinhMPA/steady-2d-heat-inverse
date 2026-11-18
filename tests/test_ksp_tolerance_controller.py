import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ksp_tolerance_controller import (  # noqa: E402
    AdaptiveKSPRTOLSettings,
    SharedKSPToleranceController,
)


class _FakeSolver:
    def __init__(self, rtol: float):
        self.rtol = float(rtol)
        self.history = [self.rtol]

    def get_ksp_rtol(self) -> float:
        return self.rtol

    def set_ksp_rtol(self, rtol: float) -> None:
        self.rtol = float(rtol)
        self.history.append(self.rtol)


class SharedKSPToleranceControllerTest(unittest.TestCase):
    def test_synchronizes_initial_rtol_across_solvers(self) -> None:
        for start_rtol in (1e-4, 1e-6):
            with self.subTest(start_rtol=start_rtol):
                fwd = _FakeSolver(start_rtol)
                adj = _FakeSolver(1e-8)
                controller = SharedKSPToleranceController((fwd, adj))

                self.assertAlmostEqual(controller.current_rtol, start_rtol)
                self.assertAlmostEqual(fwd.get_ksp_rtol(), start_rtol)
                self.assertAlmostEqual(adj.get_ksp_rtol(), start_rtol)
                self.assertEqual(controller.history, [start_rtol])

    def test_tightens_after_gradient_stall_for_multiple_starting_rtols(self) -> None:
        for start_rtol in (1e-4, 1e-6):
            with self.subTest(start_rtol=start_rtol):
                fwd = _FakeSolver(start_rtol)
                adj = _FakeSolver(start_rtol)
                controller = SharedKSPToleranceController(
                    (fwd, adj),
                    settings=AdaptiveKSPRTOLSettings(
                        initial_rtol=start_rtol,
                        min_rtol=1e-8,
                        tighten_factor=0.1,
                        stall_iterations=2,
                        stall_rel_improve=0.1,
                        rejection_iterations=3,
                        rejected_step_threshold=0.0,
                    ),
                )

                self.assertFalse(
                    controller.observe(iteration=0, gradient_norm=10.0, step_length=1.0)
                )
                self.assertFalse(
                    controller.observe(iteration=1, gradient_norm=9.5, step_length=1.0)
                )
                tightened = controller.observe(
                    iteration=2, gradient_norm=9.1, step_length=1.0
                )

                expected_rtol = max(1e-8, start_rtol * 0.1)
                self.assertTrue(tightened)
                self.assertAlmostEqual(controller.current_rtol, expected_rtol)
                self.assertAlmostEqual(fwd.get_ksp_rtol(), expected_rtol)
                self.assertAlmostEqual(adj.get_ksp_rtol(), expected_rtol)

    def test_tightens_after_repeated_small_step_lengths(self) -> None:
        start_rtol = 1e-5
        fwd = _FakeSolver(start_rtol)
        adj = _FakeSolver(start_rtol)
        controller = SharedKSPToleranceController(
            (fwd, adj),
            settings=AdaptiveKSPRTOLSettings(
                initial_rtol=start_rtol,
                min_rtol=1e-8,
                tighten_factor=0.1,
                stall_iterations=5,
                stall_rel_improve=0.0,
                rejection_iterations=2,
                rejected_step_threshold=0.2,
            ),
        )

        self.assertFalse(
            controller.observe_tao_status((0, 1.0, 10.0, 0.0, 1.0, 0))
        )
        self.assertFalse(
            controller.observe_tao_status((1, 0.9, 7.5, 0.0, 0.1, 0))
        )
        tightened = controller.observe_tao_status((2, 0.85, 6.5, 0.0, 0.05, 0))

        self.assertTrue(tightened)
        self.assertAlmostEqual(controller.current_rtol, 1e-6)
        self.assertEqual(controller.num_tightenings, 1)
        self.assertAlmostEqual(fwd.history[-1], 1e-6)
        self.assertAlmostEqual(adj.history[-1], 1e-6)


if __name__ == "__main__":
    unittest.main()
