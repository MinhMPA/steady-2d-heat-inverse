# `ksp_tolerance_controller`

The multi-fidelity mechanism: a shared adaptive KSP relative tolerance. Optimization starts
at a loose (cheap, low-fidelity) `ksp_rtol` and tightens toward `min_rtol` when the
optimizer stalls or its steps are repeatedly rejected.

Forward and adjoint solves are driven from the *same* controller, so the gradient always
comes from the same fidelity as the objective it accompanies — a mismatch there would make
the line search compare incommensurable quantities.

```{eval-rst}
.. automodule:: ksp_tolerance_controller
   :members:
   :undoc-members:
   :show-inheritance:
```
