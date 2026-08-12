# tests/test_domain_coefficient.py
# numerical imports
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CloughTocher2DInterpolator

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


def _nonlinear_table(n: int = 4) -> np.ndarray:
    """A coarse, strongly nonlinear table: the two schemes must disagree off its nodes."""
    g = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack(
        [xx.ravel(), yy.ravel(), (np.sin(4.0 * xx) * np.exp(yy)).ravel()]
    )


def test_both_interpolators_reproduce_a_linear_field(unit_square):
    """h = 1 + x is linear, so RBF (degree-1 tail) and Clough-Tocher both reproduce it."""
    domain, V = unit_square
    table = _linear_table()
    expected = 1.0 + V.tabulate_dof_coordinates()[:, 0]

    for kind in ("rbf", "ct"):
        coeff = ThermalConductivity(table, domain, V, tab_interpolator=kind)
        np.testing.assert_allclose(coeff.function.x.array, expected, atol=1e-8)


def test_tab_interpolator_actually_selects_the_scheme():
    """On a mesh finer than the table, the two schemes must produce different fields.

    A linear field cannot detect this bug -- both schemes reproduce it exactly -- so this
    test deliberately uses a coarse nonlinear table on an 8x8 mesh.
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    table = _nonlinear_table()

    rbf = ThermalConductivity(table, domain, V, tab_interpolator="rbf")
    ct = ThermalConductivity(table, domain, V, tab_interpolator="ct")

    difference = np.abs(rbf.function.x.array - ct.function.x.array).max()
    assert (
        difference > 1e-3
    ), "Both coefficients produced the same field, so tab_interpolator was ignored."

    # And the "ct" field must match SciPy's Clough-Tocher result directly.
    coords = V.tabulate_dof_coordinates()[:, :2]
    expected_ct = CloughTocher2DInterpolator(
        table[:, :2], table[:, 2], fill_value=table[:, 2].mean(), rescale=True
    )(coords)
    np.testing.assert_allclose(ct.function.x.array, expected_ct, atol=1e-12)


def test_tab_interpolator_defaults_to_rbf(unit_square):
    """The default must stay RBF -- that is the behavior every existing result relies on."""
    domain, V = unit_square
    table = _linear_table()

    default = ThermalConductivity(table, domain, V)
    explicit_rbf = ThermalConductivity(table, domain, V, tab_interpolator="rbf")

    assert default._tab_interpolator == "rbf"
    np.testing.assert_allclose(
        default.function.x.array, explicit_rbf.function.x.array, rtol=1e-12
    )


def test_unknown_tab_interpolator_raises_valueerror(unit_square):
    domain, V = unit_square

    with pytest.raises(ValueError, match="Unsupported tab_interpolator"):
        ThermalConductivity(_linear_table(), domain, V, tab_interpolator="linear")


def test_forward_solver_forwards_tab_interpolator():
    """The forward solver must expose the interpolator choice, not silently drop it."""
    from forward_solver import SteadyHeat2DForwardSolver

    fwd = SteadyHeat2DForwardSolver(
        nmesh=4, h=_linear_table(), q=1.0, tab_interpolator="ct"
    )

    assert fwd.h._tab_interpolator == "ct"
    assert fwd.q._tab_interpolator == "ct"
