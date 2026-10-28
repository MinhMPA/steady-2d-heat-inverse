"""Real finite-element checks of frozen physical heat experiments."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from dolfinx import fem
from domain_coefficient import ThermalConductivity
from heat_model import (
    FrozenHeatModel,
    HeatNumerics,
    NodalField,
    coerce_field,
    canonical_dof_indices,
)
from ._hmc_helpers import make_forward, state_arrays


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_frozen_heat_model_survives_source_and_mesh_mutation(cell):
    fwd = make_forward(cell=cell, q=2.0, bottom=315.0)
    frozen = FrozenHeatModel.from_forward(fwd)
    expected = frozen.fingerprint
    fwd.q.function.value = 9.0
    fwd.mesh.geometry.x[:, 0] *= 0.5
    clone = frozen.build_forward(prefix="hmc_test_a_", numerics=HeatNumerics())
    assert frozen.fingerprint == expected
    assert np.allclose(frozen.source, 2.0)
    assert np.allclose(frozen.boundary_values, 315.0)
    assert clone.h.function.x.array.size == 25
    assert np.isfinite(clone.solve().x.array).all()
    assert clone.mesh.comm.size == 1
    assert not np.shares_memory(clone.mesh.geometry.x, fwd.mesh.geometry.x)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_uniform_initialization_retains_all_field_unknowns(cell):
    fwd = make_forward(cell=cell, h=2.0)
    before = state_arrays(fwd)
    frozen = FrozenHeatModel.from_forward(fwd)
    clone = frozen.build_forward(prefix="hmc_test_b_", numerics=HeatNumerics())
    values = coerce_field(2.0, frozen)
    assert values.shape == (25,)
    assert np.array_equal(values, np.full(25, 2.0))
    assert isinstance(clone.h.function, fem.Function)
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(fwd)))


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_rebuilt_experiment_preserves_nonuniform_source_and_boundary(cell):
    fwd = make_forward(cell=cell, q=lambda x: 1.0 + x[0] + 2.0 * x[1])
    boundary = fem.Function(fwd.V)
    boundary.interpolate(lambda x: 310.0 + 4.0 * x[0])
    fwd.bcs = [fem.dirichletbc(boundary, fwd.bottom_dofs)]
    fwd._initialize_linear_problem("hmc_original_")
    original_T = fwd.T
    fwd.solve()
    before = state_arrays(fwd)
    frozen = FrozenHeatModel.from_forward(fwd)
    clone = frozen.build_forward(prefix="hmc_agreement_", numerics=HeatNumerics())
    np.testing.assert_allclose(
        clone.solve().x.array[canonical_dof_indices(clone.V)],
        fwd.T.x.array[canonical_dof_indices(fwd.V)],
        rtol=1e-10,
        atol=1e-10,
    )
    assert fwd.T is original_T
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(fwd)))
    assert frozen.fingerprint == FrozenHeatModel.from_forward(clone).fingerprint
    assert frozen.cells.dtype == np.int64
    for values in (
        frozen.coordinates,
        frozen.cells,
        frozen.source,
        frozen.boundary_dofs,
        frozen.boundary_values,
    ):
        assert not values.flags.writeable
    boundary.x.array[:] = 0.0
    assert np.all(frozen.boundary_values >= 310.0)


def test_field_conversion_uses_physical_coordinates_and_accepts_log_means():
    fwd = make_forward()
    model = FrozenHeatModel.from_forward(fwd)
    expected = -2.0 + model.coordinates[:, 0] + 3.0 * model.coordinates[:, 1]
    np.testing.assert_allclose(
        coerce_field(lambda x: -2.0 + x[0] + 3.0 * x[1], model), expected
    )
    values = fem.Function(fwd.V)
    values.interpolate(lambda x: -2.0 + x[0] + 3.0 * x[1])
    # Interpolation evaluates this analytic field with floating-point arithmetic.
    np.testing.assert_allclose(
        coerce_field(values, model), expected, atol=8 * np.finfo(float).eps
    )
    permutation = np.random.default_rng(21).permutation(25)
    nodal = NodalField(expected[permutation], model.coordinates[permutation])
    converted = coerce_field(nodal, model)
    np.testing.assert_array_equal(converted, expected)
    converted[:] = 100.0
    np.testing.assert_array_equal(nodal.values, expected[permutation])
    np.testing.assert_array_equal(coerce_field(-2.0, model), np.full(25, -2.0))
    with pytest.raises(TypeError, match="coordinates|NodalField"):
        coerce_field(expected, model)
    with pytest.raises(ValueError, match="coordinates|mesh"):
        coerce_field(NodalField(expected, model.coordinates + 0.01), model)
    with pytest.raises(ValueError, match="ambiguous|unique|duplicate"):
        coerce_field(NodalField(expected, np.zeros((25, 2))), model)


@pytest.mark.parametrize(
    "bad",
    [
        1.0 + 0j,
        np.nan,
        np.inf,
        NodalField(np.ones(25, dtype=complex), np.zeros((25, 2))),
    ],
)
def test_field_conversion_rejects_nonreal_or_nonfinite_values(bad):
    model = FrozenHeatModel.from_forward(make_forward())
    with pytest.raises((TypeError, ValueError), match="real|finite"):
        coerce_field(bad, model)


@pytest.mark.parametrize("kind", ["source", "boundary"])
def test_nonfinite_physical_inputs_are_rejected(kind):
    fwd = make_forward()
    if kind == "source":
        fwd.q.function.value = np.nan
    else:
        fwd.bcs = [fem.dirichletbc(np.float64(np.inf), fwd.bottom_dofs, fwd.V)]
    with pytest.raises(ValueError, match="finite"):
        FrozenHeatModel.from_forward(fwd)


def test_physical_fingerprint_excludes_current_conductivity_and_temperature():
    fwd = make_forward(h=lambda x: 1.0 + x[0])
    first = FrozenHeatModel.from_forward(fwd)
    fwd.h.function.x.array[:] = 3.0
    fwd.solve()
    assert FrozenHeatModel.from_forward(fwd).fingerprint == first.fingerprint
    fwd.q.function.value = 2.0
    assert FrozenHeatModel.from_forward(fwd).fingerprint != first.fingerprint


def test_scalar_p1_function_is_accepted_but_other_spaces_are_rejected():
    fwd = make_forward()
    values = fem.Function(fwd.V)
    values.x.array[:] = 2.0
    assert ThermalConductivity(values, fwd.mesh, fwd.V).function is values
    other = make_forward()
    with pytest.raises(ValueError, match="space|mesh"):
        ThermalConductivity(values, other.mesh, other.V)
    fwd.V = fem.functionspace(fwd.mesh, ("Lagrange", 2))
    with pytest.raises(ValueError, match="P1|Q1|degree|element"):
        FrozenHeatModel.from_forward(fwd)


def test_workspaces_have_independent_fields_and_zero_initial_guesses():
    frozen = FrozenHeatModel.from_forward(make_forward())
    a = frozen.build_forward(prefix="hmc_workspace_a_", numerics=HeatNumerics())
    b = frozen.build_forward(prefix="hmc_workspace_b_", numerics=HeatNumerics())
    assert a.problem.solver.getOptionsPrefix() != b.problem.solver.getOptionsPrefix()
    assert not a.problem.solver.getInitialGuessNonzero()
    expected = b.solve().x.array.copy()
    a.h.function.x.array[:] = 5.0
    a.solve()
    np.testing.assert_array_equal(b.T.x.array, expected)
    b.T.x.array[:] = 1.0e9
    np.testing.assert_array_equal(b.solve().x.array, expected)
    with pytest.raises(ValueError, match="positive"):
        a.from_discrete_data(
            domain=a.mesh,
            h_values=np.zeros(25),
            q_values=np.ones(25),
            boundary_dofs=a.bottom_dofs,
            boundary_values=np.full(5, 300.0),
            petsc_opts={},
            prefix="hmc_invalid_",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtol": np.nan},
        {"rtol": 0.0},
        {"atol": -1.0},
        {"atol": np.inf},
        {"max_it": 0},
        {"max_it": 2.5},
        {"ksp_type": ""},
        {"pc_type": ""},
    ],
)
def test_heat_numerics_rejects_invalid_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        HeatNumerics(**kwargs)


def test_heat_numerics_is_frozen_and_applied():
    numerics = HeatNumerics(rtol=1e-11, atol=1e-13, max_it=1234)
    with pytest.raises(FrozenInstanceError):
        numerics.rtol = 0.1
    clone = FrozenHeatModel.from_forward(make_forward()).build_forward(
        prefix="hmc_numerics_", numerics=numerics
    )
    rtol, atol, _, max_it = clone.get_ksp_tolerances()
    assert (rtol, atol, max_it) == (1e-11, 1e-13, 1234)
    assert clone.petsc_opts["ksp_error_if_not_converged"] is True


def test_multirank_experiment_rejected_before_new_forms():
    code = """
from mpi4py import MPI
from dolfinx import mesh, fem
from types import SimpleNamespace
from heat_model import FrozenHeatModel
domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
fwd = SimpleNamespace(mesh=domain)
try:
    FrozenHeatModel.from_forward(fwd)
except ValueError as exc:
    assert "single" in str(exc).lower() or "rank" in str(exc).lower()
else:
    raise AssertionError("multi-rank heat experiment accepted")
"""
    result = subprocess.run(
        [
            str(Path(sys.prefix) / "bin" / "mpiexec"),
            "-n",
            "2",
            sys.executable,
            "-c",
            code,
        ],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_canonical_order_uses_physical_vertices_despite_coordinate_roundoff(cell):
    fwd = make_forward(cell=cell)
    physical = fwd.mesh.geometry.x[:, :2]
    physical_order = np.lexsort((physical[:, 1], physical[:, 0]))
    native_to_geometry = np.empty(len(physical), dtype=int)
    for dofs, vertices in zip(fwd.V.dofmap.list, fwd.mesh.geometry.dofmaps[0]):
        native_to_geometry[dofs] = vertices
    order = canonical_dof_indices(fwd.V)
    np.testing.assert_array_equal(
        physical[native_to_geometry[order]], physical[physical_order]
    )


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_nodal_field_accepts_native_tabulated_coordinates_with_roundoff(cell):
    fwd = make_forward(cell=cell)
    model = FrozenHeatModel.from_forward(fwd)
    values = fem.Function(fwd.V)
    values.x.array[:] = np.arange(len(values.x.array)) + 0.125
    labels = fwd.V.tabulate_dof_coordinates()[:, :2]
    expected = values.x.array[canonical_dof_indices(fwd.V)].copy()
    permutation = np.random.default_rng(57).permutation(len(expected))
    actual = coerce_field(
        NodalField(values.x.array[permutation], labels[permutation]), model
    )
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="coordinates|mesh"):
        coerce_field(NodalField(values.x.array, labels + 1e-8), model)
