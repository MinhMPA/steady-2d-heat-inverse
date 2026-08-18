# `benchmark_utils`

Harness for comparing optimization strategies on the same inverse problem. A
{py:class}`~benchmark_utils.BenchmarkCase` fixes the physics (true and initial
conductivity, mesh, noise, regularization); a
{py:class}`~benchmark_utils.BenchmarkMethod` fixes the solver strategy (fixed or adaptive
KSP tolerance, TAO type, tolerances). `run_benchmark_case` pairs them and returns the
solution together with a per-iteration trace and a summary of KSP work.

See [`notebooks/MultiFidelitySolve.ipynb`](https://github.com/MinhMPA/steady-2d-heat-inverse/blob/master/notebooks/MultiFidelitySolve.ipynb)
for a worked comparison of a fixed-tolerance baseline against an adaptive schedule.

```{eval-rst}
.. automodule:: benchmark_utils
   :members:
   :undoc-members:
   :show-inheritance:
```
