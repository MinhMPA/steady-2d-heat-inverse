"""Independent covariance and dual-map checks for the full nodal Gaussian prior."""

import json

import numpy as np
import pytest
from dolfinx import fem

from conductivity_prior import (
    ConductivityPrior,
    GaussianLogConductivity,
    InvertibleConductivityPrior,
)
from heat_model import FrozenHeatModel, HeatNumerics, NodalField
from ._hmc_helpers import make_forward, state_arrays


def _dense_element_matrices(model):
    """Integrate P1/Q1 basis functions directly, independently of UFL assembly."""
    count = len(model.coordinates)
    mass, stiffness = np.zeros((count, count)), np.zeros((count, count))
    for nodes in model.cells:
        xy = model.coordinates[nodes]
        if model.cell_type == "triangle":
            jacobian = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
            area = abs(np.linalg.det(jacobian)) / 2.0
            gradients = np.array(
                [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
            ) @ np.linalg.inv(jacobian)
            local_mass = area / 12.0 * (np.ones((3, 3)) + np.eye(3))
            local_stiffness = area * gradients @ gradients.T
        else:
            local_mass, local_stiffness = np.zeros((4, 4)), np.zeros((4, 4))
            for xi in (0.5 - 1.0 / np.sqrt(12.0), 0.5 + 1.0 / np.sqrt(12.0)):
                for eta in (0.5 - 1.0 / np.sqrt(12.0), 0.5 + 1.0 / np.sqrt(12.0)):
                    basis = np.array(
                        [(1 - xi) * (1 - eta), xi * (1 - eta), (1 - xi) * eta, xi * eta]
                    )
                    derivatives = np.array(
                        [
                            [-(1 - eta), -(1 - xi)],
                            [1 - eta, -xi],
                            [-eta, 1 - xi],
                            [eta, xi],
                        ]
                    )
                    jacobian = xy.T @ derivatives
                    gradients = derivatives @ np.linalg.inv(jacobian)
                    weight = abs(np.linalg.det(jacobian)) / 4.0
                    local_mass += weight * np.outer(basis, basis)
                    local_stiffness += weight * gradients @ gradients.T
        mass[np.ix_(nodes, nodes)] += local_mass
        stiffness[np.ix_(nodes, nodes)] += local_stiffness
    return mass, stiffness


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
@pytest.mark.parametrize("rebuilt", [False, True])
def test_full_prior_map_inverse_and_transpose(cell, rebuilt):
    fwd = make_forward(nmesh=2, cell=cell)
    model = FrozenHeatModel.from_forward(fwd)
    V = fwd.V
    if rebuilt:
        V = model.build_forward(prefix="hmc_prior_", numerics=HeatNumerics()).V
    before = state_arrays(fwd)
    prior = GaussianLogConductivity(mu=0.3, kappa=2.0, tau=1.5, h_ref=2.0).bind(
        model, V
    )
    z = np.linspace(-0.4, 0.4, prior.dimension)
    assert isinstance(prior, ConductivityPrior)
    assert isinstance(prior, InvertibleConductivityPrior)
    assert prior.dimension == 9
    assert prior.coordinate_schema == {"z": (9,)}
    assert prior.coordinate_law == "standard_normal"
    np.testing.assert_allclose(prior.from_h(prior.to_h(z)), z, atol=2e-12)
    baseline = prior.to_log_h(np.zeros(9))
    J = np.column_stack([prior.to_log_h(e) - baseline for e in np.eye(9)])
    v = np.arange(9, dtype=float)
    np.testing.assert_allclose(prior.pullback(z, v), J.T @ v, atol=2e-12)
    assert prior.energy(z) == pytest.approx(0.5 * z @ z)
    assert np.array_equal(prior.grad_energy(z), z)
    assert not np.shares_memory(prior.grad_energy(z), z)
    mass, stiffness = _dense_element_matrices(model)
    A = 4.0 * mass + stiffness
    lumped = mass @ np.ones(9)
    expected_J = np.linalg.solve(A, np.diag(np.sqrt(lumped))) / 1.5
    np.testing.assert_allclose(J, expected_J, rtol=1e-11, atol=2e-14)
    covariance = np.linalg.solve(A, np.diag(lumped)) @ np.linalg.inv(A.T) / 1.5**2
    np.testing.assert_allclose(J @ J.T, covariance, rtol=1e-11, atol=2e-14)
    # This direction maps exactly to a constant log-field when K has natural boundaries.
    constant_z = 1.5 * 4.0 * np.sqrt(lumped)
    np.testing.assert_allclose(
        prior.to_log_h(constant_z) - baseline, np.ones(9), atol=2e-12
    )
    assert np.ones(9) @ covariance @ np.ones(9) > 0.0
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(fwd)))


@pytest.mark.parametrize("kind", ["scalar", "callable", "nodal", "function"])
@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_mean_fields_are_canonical_and_frozen(kind, cell):
    fwd = make_forward(nmesh=2, cell=cell)
    model = FrozenHeatModel.from_forward(fwd)
    expected = -2.0 + model.coordinates[:, 0] + 3.0 * model.coordinates[:, 1]
    if kind == "scalar":
        supplied, expected = -2.0, np.full(9, -2.0)
    elif kind == "callable":
        supplied = lambda x: -2.0 + x[0] + 3.0 * x[1]
    elif kind == "nodal":
        order = np.random.default_rng(31).permutation(9)
        supplied = NodalField(expected[order].copy(), model.coordinates[order].copy())
    else:
        supplied = fem.Function(fwd.V)
        supplied.interpolate(lambda x: -2.0 + x[0] + 3.0 * x[1])
    prior = GaussianLogConductivity(mu=supplied, kappa=2.0, tau=1.5, h_ref=2.0).bind(
        model, fwd.V
    )
    first_record = prior.record()
    if kind == "nodal":
        supplied.values[:] = 99.0
        supplied.coordinates[:] = -1.0
    elif kind == "function":
        supplied.x.array[:] = 99.0
    np.testing.assert_allclose(prior.to_log_h(np.zeros(9)), expected, atol=2e-15)
    assert prior.record() == first_record
    assert json.loads(json.dumps(prior.record())) == prior.record()


@pytest.mark.parametrize("parameter", ["kappa", "tau", "h_ref"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, 1.0 + 0j, [1.0]])
def test_prior_rejects_invalid_physical_parameters(parameter, value):
    parameters = dict(mu=0.0, kappa=2.0, tau=1.5, h_ref=2.0)
    parameters[parameter] = value
    with pytest.raises((TypeError, ValueError), match=parameter):
        GaussianLogConductivity(**parameters)


@pytest.mark.parametrize("mu", [np.nan, np.inf, 1.0 + 0j])
def test_prior_rejects_invalid_evaluated_means(mu):
    fwd = make_forward(nmesh=2)
    model = FrozenHeatModel.from_forward(fwd)
    with pytest.raises((TypeError, ValueError), match="real|finite"):
        GaussianLogConductivity(mu=mu, kappa=2.0, tau=1.5, h_ref=2.0).bind(model, fwd.V)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, 1.0 + 0j])
def test_inverse_rejects_nonpositive_or_nonfinite_conductivity(bad):
    fwd = make_forward(nmesh=2)
    prior = GaussianLogConductivity(mu=0.0, kappa=2.0, tau=1.5, h_ref=2.0).bind(
        FrozenHeatModel.from_forward(fwd), fwd.V
    )
    values = np.ones(9, dtype=complex if np.iscomplexobj(bad) else float)
    values[3] = bad
    with pytest.raises((TypeError, ValueError), match="positive|finite|real"):
        prior.from_h(values)


@pytest.mark.parametrize(
    "method", ["to_h", "to_log_h", "from_h", "energy", "grad_energy"]
)
def test_prior_rejects_wrong_coordinate_shapes(method):
    fwd = make_forward(nmesh=2)
    prior = GaussianLogConductivity(mu=0.0, kappa=2.0, tau=1.5, h_ref=2.0).bind(
        FrozenHeatModel.from_forward(fwd), fwd.V
    )
    with pytest.raises(ValueError, match="shape|dimension"):
        getattr(prior, method)(np.ones(8))
    with pytest.raises(ValueError, match="shape|dimension"):
        prior.pullback(np.ones(9), np.ones(8))


def test_prior_record_tracks_physics_and_excludes_initial_state():
    fwd = make_forward(nmesh=2)
    model = FrozenHeatModel.from_forward(fwd)
    specification = GaussianLogConductivity(mu=0.3, kappa=2.0, tau=1.5, h_ref=2.0)
    prior = specification.bind(model, fwd.V)
    record = prior.record()
    assert record["fingerprint"]
    for z in (np.zeros(9), np.ones(9)):
        prior.from_h(prior.to_h(z))
    fwd.h.function.value = 7.0
    fwd.T.x.array[:] = 99.0
    assert (
        specification.bind(FrozenHeatModel.from_forward(fwd), fwd.V).record() == record
    )
    for change in ({"mu": 0.4}, {"kappa": 3.0}, {"tau": 2.0}, {"h_ref": 3.0}):
        parameters = dict(mu=0.3, kappa=2.0, tau=1.5, h_ref=2.0)
        parameters.update(change)
        changed = GaussianLogConductivity(**parameters).bind(model, fwd.V)
        assert changed.record()["fingerprint"] != record["fingerprint"]
    record["fingerprint"] = "changed"
    assert prior.record()["fingerprint"] != "changed"


@pytest.mark.parametrize(
    "mean, reference, conductivity",
    [(-750.0, 1e300, 1.9016849634750064e-26), (750.0, 1e-300, 5.258494541454804e25)],
)
def test_reference_scale_preserves_representable_conductivity(
    mean, reference, conductivity
):
    fwd = make_forward(nmesh=2)
    prior = GaussianLogConductivity(mu=mean, kappa=2.0, tau=1.5, h_ref=reference).bind(
        FrozenHeatModel.from_forward(fwd), fwd.V
    )
    # Physical h_ref * exp(mean), evaluated independently at 70-digit precision.
    expected = np.full(9, conductivity)
    np.testing.assert_allclose(prior.to_h(np.zeros(9)), expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        prior.to_h(prior.from_h(expected)), expected, rtol=1e-12, atol=0.0
    )


def test_prior_does_not_clip_exponential_overflow():
    fwd = make_forward(nmesh=2)
    prior = GaussianLogConductivity(mu=0.0, kappa=2.0, tau=1.5, h_ref=2.0).bind(
        FrozenHeatModel.from_forward(fwd), fwd.V
    )
    assert np.isposinf(prior.to_h(np.full(9, 1e6))).any()


def test_binding_rejects_a_different_physical_mesh():
    fwd = make_forward(nmesh=2)
    model = FrozenHeatModel.from_forward(fwd)
    other = make_forward(nmesh=2)
    other.mesh.geometry.x[:, 0] *= 0.5
    with pytest.raises(ValueError, match="mesh|coordinates"):
        GaussianLogConductivity(mu=0.0, kappa=2.0, tau=1.5, h_ref=2.0).bind(
            model, other.V
        )
