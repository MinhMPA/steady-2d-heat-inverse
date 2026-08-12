# tests/test_domain_coefficient.py
# numerical imports
import numpy as np
import pandas as pd
import pytest

# mpi imports
from mpi4py import MPI

# dolfinx imports
from dolfinx import mesh, fem

# local imports
from domain_coefficient import ThermalConductivity


@pytest.fixture
def unit_square():
    """A small quadrilateral unit-square mesh and its P1 Lagrange space."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V


def _linear_table(n: int = 5) -> np.ndarray:
    """Tabulate h(x,y) = 1 + x on a regular (n,n) grid over the unit square."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), (1.0 + xx).ravel()])


def test_dataframe_input_matches_ndarray_input(unit_square):
    """A (x|y|value) DataFrame must interpolate identically to the equivalent (N,3) array."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame({"x": table[:, 0], "y": table[:, 1], "value": table[:, 2]})

    from_array = ThermalConductivity(table, domain, V)
    from_frame = ThermalConductivity(frame, domain, V)

    np.testing.assert_allclose(
        from_frame.function.x.array, from_array.function.x.array, rtol=1e-12
    )


def test_dataframe_column_names_are_case_insensitive(unit_square):
    """Column matching lower-cases the header, so (X|Y|Value) is accepted."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame({"X": table[:, 0], "Y": table[:, 1], "Value": table[:, 2]})

    from_array = ThermalConductivity(table, domain, V)
    from_frame = ThermalConductivity(frame, domain, V)

    np.testing.assert_allclose(
        from_frame.function.x.array, from_array.function.x.array, rtol=1e-12
    )


def test_dataframe_missing_column_raises_valueerror(unit_square):
    """A frame without a `value` column must raise ValueError, not KeyError."""
    domain, V = unit_square
    table = _linear_table()
    frame = pd.DataFrame({"x": table[:, 0], "y": table[:, 1], "h": table[:, 2]})

    with pytest.raises(ValueError, match=r"\(x\|y\|value\)"):
        ThermalConductivity(frame, domain, V)
