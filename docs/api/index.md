# API reference

The [conductivity posterior reference](heat_inference.md) documents fixed physical
observations, the shared optimization/sampling posterior, complete states, saved
fields, convergence reports and trajectory controls. The remaining pages describe
the original heat, adjoint and TAO interfaces; their legacy reconstruction objective
is distinguished in the [theory](../theory.md).

Signatures and detailed member descriptions are generated from module docstrings.
Remember that these are **flat top-level modules**
(see {doc}`../installation`), so `forward_solver` — not `src.forward_solver`.

```{toctree}
:maxdepth: 2

heat_inference
forward_solver
adjoint_solver
tao_solver
domain_coefficient
ksp_tolerance_controller
benchmark_utils
plotting_utils
```
