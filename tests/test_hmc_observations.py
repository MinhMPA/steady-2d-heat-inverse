"""Physical sensor predictions and independent Gaussian temperature errors."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from dolfinx import fem, geometry
from scipy.sparse import isspmatrix_csr

from temperature_observations import SensorOperator, TemperatureObservations
from ._hmc_helpers import make_forward

SENSOR_POINTS = np.array(
    [
        [0.2, 0.3],
        [0.2, 0.3],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.25],
        [0.0, 0.65],
        [0.7, 0.0],
        [1.0, 0.4],
        [0.6, 1.0],
    ]
)


def dolfinx_readings(field, points):
    """Use DOLFINx's own field evaluation, selecting any containing cell."""
    domain = field.function_space.mesh
    xyz = np.column_stack((points, np.zeros(len(points))))
    tree = geometry.bb_tree(domain, domain.topology.dim, padding=1e-14)
    candidates = geometry.compute_collisions_points(tree, xyz)
    collisions = geometry.compute_colliding_cells(domain, candidates, xyz)
    assert all(len(collisions.links(i)) for i in range(len(points)))
    cells = np.array(
        [collisions.links(i)[-1] for i in range(len(points))], dtype=np.int32
    )
    return field.eval(xyz, cells).reshape(-1)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_sensor_prediction_and_transpose_agree_with_physical_field(cell):
    fwd = make_forward(cell=cell)
    points = SENSOR_POINTS.copy()
    operator = SensorOperator(fwd.V, points)
    coordinates = fwd.V.tabulate_dof_coordinates()
    temperature = 300.0 + 2 * coordinates[:, 0] - 3 * coordinates[:, 1]
    expected = 300.0 + 2 * points[:, 0] - 3 * points[:, 1]
    assert isspmatrix_csr(operator.matrix)
    assert operator.matrix.shape == (len(points), len(temperature))
    np.testing.assert_allclose(
        operator.predict(temperature), expected, rtol=0, atol=1e-12
    )
    field = fem.Function(fwd.V)
    field.x.array[:] = temperature
    field.x.scatter_forward()
    np.testing.assert_allclose(
        operator.predict(temperature),
        dolfinx_readings(field, points),
        rtol=0,
        atol=1e-12,
    )
    weights = np.arange(len(points)) - 2.5
    assert np.isclose(
        weights @ operator.predict(temperature),
        operator.transpose(weights) @ temperature,
        rtol=0,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        operator.matrix.sum(axis=1), np.ones((len(points), 1)), rtol=0, atol=1e-14
    )
    # Duplicate instruments remain two independent entries, including on walls.
    np.testing.assert_array_equal(
        operator.matrix.getrow(0).toarray(), operator.matrix.getrow(1).toarray()
    )
    points[:] = -100
    np.testing.assert_allclose(
        operator.predict(temperature), expected, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_sensor_basis_handles_arbitrary_native_nodal_temperature(cell):
    fwd = make_forward(cell=cell)
    field = fem.Function(fwd.V)
    field.x.array[:] = 300 + np.random.default_rng(11).normal(size=len(field.x.array))
    field.x.scatter_forward()
    points = np.vstack((SENSOR_POINTS, np.random.default_rng(31).uniform(size=(30, 2))))
    operator = SensorOperator(fwd.V, points)
    np.testing.assert_allclose(
        operator.predict(field.x.array),
        dolfinx_readings(field, points),
        rtol=0,
        atol=1e-12,
    )
    weights = np.random.default_rng(18).normal(size=len(points))
    assert np.isclose(
        weights @ operator.predict(field.x.array),
        operator.transpose(weights) @ field.x.array,
        rtol=0,
        atol=1e-11,
    )


def test_observations_own_immutable_points_values_and_noise():
    points = SENSOR_POINTS.copy()
    values = np.arange(len(points)) + 300.0
    sigma = np.linspace(0.001, 0.009, len(points))
    observations = TemperatureObservations(points, values, sigma)
    points[:] = -100
    values[:] = -100
    sigma[:] = -100
    np.testing.assert_array_equal(observations.points, SENSOR_POINTS)
    np.testing.assert_array_equal(observations.values, np.arange(len(points)) + 300.0)
    np.testing.assert_array_equal(
        observations.sigma, np.linspace(0.001, 0.009, len(points))
    )
    for array in (observations.points, observations.values, observations.sigma):
        with pytest.raises(ValueError):
            array.flat[0] = 1.0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        observations.sigma = 0.01
    scalar = np.array(0.001)
    uniform = TemperatureObservations(
        SENSOR_POINTS, np.full(len(points), 300.0), scalar
    )
    scalar[...] = -1
    np.testing.assert_allclose(uniform.sigma, 0.001, rtol=0, atol=0)


@pytest.mark.parametrize(
    "points,values,sigma,message",
    [
        (np.empty((0, 2)), [], 1.0, "sensor"),
        ([[0.2, 0.3], [np.nan, 0.2]], [300, 301], 1.0, "sensor.*1"),
        ([[0.2, 0.3], [0.3, np.inf]], [300, 301], 1.0, "sensor.*1"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, np.nan], 1.0, "sensor.*1"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, np.inf], 1.0, "sensor.*1"),
        ([[0.2, 0.3]], [300, 301], 1.0, "sensor|length|shape"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, 301], [1.0], "sensor|length|shape"),
        ([[0.2, 0.3]], [300], 0.0, "sensor.*0"),
        ([[0.2, 0.3]], [300], -1.0, "sensor.*0"),
        ([[0.2, 0.3]], [300], np.nan, "sensor.*0"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, 301], [1.0, 0.0], "sensor.*1"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, 301], [1.0, -1.0], "sensor.*1"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, 301], [1.0, np.nan], "sensor.*1"),
        ([[0.2, 0.3], [0.3, 0.2]], [300, 301], [1.0, np.inf], "sensor.*1"),
    ],
)
def test_invalid_observations_identify_offending_sensor(points, values, sigma, message):
    with pytest.raises(ValueError, match=message):
        TemperatureObservations(points, values, sigma)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
@pytest.mark.parametrize(
    "outside", [[-0.01, 0.3], [1.01, 0.5], [0.5, -0.01], [0.5, 1.01], [1.0 + 1e-8, 0.4]]
)
def test_sensor_operator_rejects_extrapolation_with_sensor_index(cell, outside):
    fwd = make_forward(cell=cell)
    with pytest.raises(ValueError, match="sensor.*1"):
        SensorOperator(fwd.V, [[0.2, 0.3], outside])


@pytest.mark.parametrize(
    "points,message",
    [
        (np.empty((0, 2)), "sensor"),
        ([[0.2, 0.3], [np.nan, 0.4]], "sensor.*1"),
        ([[0.2, 0.3], [0.4, np.inf]], "sensor.*1"),
        ([[0.2, 0.3, 0]], "shape|sensor"),
    ],
)
def test_sensor_operator_rejects_invalid_sensor_coordinates(points, message):
    with pytest.raises(ValueError, match=message):
        SensorOperator(make_forward().V, points)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_sensor_permutation_preserves_gaussian_misfit_and_force(cell):
    fwd = make_forward(cell=cell)
    points = SENSOR_POINTS.copy()
    values = np.arange(len(points)) + 299.5
    sigma = np.linspace(0.01, 0.03, len(points))
    observations = TemperatureObservations(points, values, sigma)
    operator = SensorOperator(fwd.V, observations.points)
    native = np.random.default_rng(99).normal(size=operator.matrix.shape[1]) + 300
    residual = operator.predict(native) - observations.values
    misfit = 0.5 * np.sum((residual / observations.sigma) ** 2)
    force = operator.transpose(residual / observations.sigma**2)
    order = np.random.default_rng(45).permutation(len(points))
    permuted = TemperatureObservations(points[order], values[order], sigma[order])
    shuffled = SensorOperator(fwd.V, permuted.points)
    np.testing.assert_allclose(
        shuffled.matrix.toarray(), operator.matrix.toarray()[order], rtol=0, atol=0
    )
    residual = shuffled.predict(native) - permuted.values
    assert np.isclose(
        0.5 * np.sum((residual / permuted.sigma) ** 2), misfit, rtol=1e-14
    )
    np.testing.assert_allclose(
        shuffled.transpose(residual / permuted.sigma**2), force, rtol=1e-14, atol=1e-9
    )


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_nodal_observations_preserve_original_noise_when_inference_mesh_changes(cell):
    original = make_forward(nmesh=6, cell=cell)
    field = fem.Function(original.V)
    noisy_values = 300 + np.random.default_rng(4).normal(size=len(field.x.array))
    field.x.array[:] = noisy_values
    field.x.scatter_forward()
    sigma = np.linspace(0.001, 0.002, len(noisy_values))
    observations = TemperatureObservations.from_function(field, sigma=sigma)
    expected_points = original.V.tabulate_dof_coordinates()[:, :2].copy()
    np.testing.assert_allclose(observations.points, expected_points, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(observations.values, noisy_values)
    np.testing.assert_array_equal(observations.sigma, sigma)
    # Reading on the original nodes must reproduce each original noisy datum.
    np.testing.assert_allclose(
        SensorOperator(original.V, observations.points).predict(noisy_values),
        noisy_values,
        rtol=0,
        atol=1e-12,
    )
    field.x.array[:] = -100
    inference = make_forward(nmesh=3, cell=cell)
    operator = SensorOperator(inference.V, observations.points)
    assert operator.matrix.shape == (49, 16)
    np.testing.assert_array_equal(observations.values, noisy_values)
    np.testing.assert_allclose(observations.points, expected_points, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(observations.sigma, sigma)
    coords = inference.V.tabulate_dof_coordinates()
    np.testing.assert_allclose(
        operator.predict(300 + 2 * coords[:, 0] - 3 * coords[:, 1]),
        300 + 2 * observations.points[:, 0] - 3 * observations.points[:, 1],
        rtol=0,
        atol=1e-12,
    )
