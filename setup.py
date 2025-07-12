# setup.py
from setuptools import setup, find_packages

setup(
    name="steady-2d-heat-inverse",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "mpi4py",
        "petsc4py",
        "fenics-basix",
        "fenics-ufl",
        "pyvista",
    ],
    python_requires=">=3.11",
)
