# steady-2d-heat-inverse

1. Solve the 2D steady-state heat equation.
2. Solve the inverse problem to infer the position-dependent thermal conductivity

## Installation

```
conda env create -n steady-2d-heat-inverse -f environment.yml
conda activate steady-2d-heat-inverse
pip install -e .
```

## Usage

- The [notebooks/ForwardSolve.ipynb](notebooks/ForwardSolve.ipynb) notebook demonstrates how to solve the forward problem.
- The [notebooks/InverseSolve.ipynb](notebooks/ForwardSolve.ipynb) notebook demonstrates how to solve the inverse problem.

## References

- [1] Hans P. Langtangen and Kent-Andre Mardal, ["Introduction to Numerical Methods for Variational Problems"](https://hplgit.github.io/fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf)
- [2] Hans P. Langtangen and Anders Logg (adapted by Jørgen S. Dokken) ["The FEniCS tutorial"](https://jsdokken.com/dolfinx-tutorial/)
- [3] Hans P. Langtangen, ["Approximation of Functions"](https://hplgit.github.io/num-methods-for-PDEs/doc/pub/approx/pdf/approx-4print.pdf)
- [4] Andrew M. Bradley, ["PDE-constrained optimization and the adjoint method"](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)

