"""Fixed physical temperature measurements and finite-element sensor readings."""

# type imports
from dataclasses import dataclass

# numerical imports
import numpy as np
from scipy.sparse import csr_matrix

# dolfinx imports
from dolfinx import fem, geometry

# local imports
from heat_model import _immutable_copy, _physical_dof_coordinates, _validate_space


def _real_sensor_array(values, name):
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} at sensor 0 must be real.")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real sensor values.") from exc
    invalid = np.argwhere(~np.isfinite(array))
    if invalid.size or (array.ndim == 0 and not np.isfinite(array)):
        index = int(invalid[0, 0]) if array.ndim else 0
        raise ValueError(f"{name} at sensor {index} must be finite.")
    return array


def _sensor_points(points):
    points = _real_sensor_array(points, "Coordinates")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Sensor coordinates must have shape (S, 2).")
    if len(points) == 0:
        raise ValueError("At least one temperature sensor is required.")
    return points


def _sensor_vector(values, count, name):
    values = _real_sensor_array(values, name)
    if values.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), one value per sensor.")
    return values


@dataclass(frozen=True)
class TemperatureObservations:
    r"""Immutable measurements with independent, known Gaussian errors.

    Parameters
    ----------
    points : numpy.ndarray
        Physical sensor coordinates, shape (S, 2), in measurement order.
    values : numpy.ndarray
        Measured temperatures in kelvin, shape (S,).
    sigma : float or numpy.ndarray
        Positive error standard deviation in kelvin, common or shape (S,).
    """

    points: np.ndarray
    values: np.ndarray
    sigma: float | np.ndarray

    def __post_init__(self):
        points = _sensor_points(self.points)
        values = _sensor_vector(self.values, len(points), "Temperature")
        sigma = _real_sensor_array(self.sigma, "Noise standard deviation")
        if sigma.ndim != 0 and sigma.shape != (len(points),):
            raise ValueError(
                "Noise standard deviations must be scalar or have shape "
                f"({len(points)},), one per sensor."
            )
        if np.any(sigma <= 0):
            index = int(np.flatnonzero(sigma <= 0)[0])
            raise ValueError(
                f"Noise standard deviation at sensor {index} must be positive."
            )
        object.__setattr__(self, "points", _immutable_copy(points, np.float64))
        object.__setattr__(self, "values", _immutable_copy(values, np.float64))
        object.__setattr__(
            self,
            "sigma",
            float(sigma) if sigma.ndim == 0 else _immutable_copy(sigma, np.float64),
        )

    @classmethod
    def from_function(cls, T_obs: fem.Function, *, sigma) -> "TemperatureObservations":
        r"""Copy original nodal measurements and their physical positions.

        Parameters
        ----------
        T_obs : dolfinx.fem.Function
            Original measured scalar P1/Q1 temperature, on one MPI rank.
        sigma : float or numpy.ndarray
            Known positive measurement errors, in original native nodal order.

        Returns
        -------
        TemperatureObservations
            Independent measurements retaining every original value and node.
        """
        if not isinstance(T_obs, fem.Function):
            raise ValueError("Nodal temperature observations require a FEM Function.")
        return cls(
            _physical_dof_coordinates(T_obs.function_space), T_obs.x.array, sigma
        )


class SensorOperator:
    r"""Map native temperature nodes to fixed physical sensor positions.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Scalar continuous P1/Q1 temperature space on one MPI rank.
    points : numpy.ndarray
        Physical sensor coordinates, shape (S, 2), retaining duplicate sensors.
    """

    def __init__(self, V, points):
        _validate_space(V)
        points = _sensor_points(points)
        domain = V.mesh
        geometry_data = domain.geometry
        dtype = geometry_data.x.dtype
        xyz = np.zeros((len(points), 3), dtype=dtype)
        xyz[:, :2] = points
        # Padding only protects shared walls/vertices from roundoff. Cell
        # membership is checked again in reference coordinates below, so the
        # collision-search tolerance cannot permit physical extrapolation.
        scale = max(1.0, float(np.max(np.abs(geometry_data.x))))
        tolerance = 64 * np.finfo(dtype).eps
        tree = geometry.bb_tree(domain, domain.topology.dim, padding=tolerance * scale)
        candidates = geometry.compute_collisions_points(tree, xyz)
        collisions = geometry.compute_colliding_cells(domain, candidates, xyz)
        coordinate_element = geometry_data.cmaps[0]
        basis = V.element.basix_element
        triangle = domain.topology.cell_name() == "triangle"
        rows, columns, entries = [], [], []
        for index, point in enumerate(points):
            # A shared edge or vertex has several containing cells. Choosing
            # the lowest native cell index makes the construction repeatable;
            # continuity makes its reading independent of this choice.
            for cell in sorted(collisions.links(index)):
                cell_geometry = geometry_data.x[
                    geometry_data.dofmaps[0][cell], : geometry_data.dim
                ]
                reference = coordinate_element.pull_back(
                    np.asarray(point[None, :], dtype=dtype), cell_geometry
                )
                inside = np.all(reference >= -tolerance)
                if triangle:
                    inside = inside and np.sum(reference) <= 1 + tolerance
                else:
                    inside = inside and np.all(reference <= 1 + tolerance)
                if not inside:
                    continue
                local_basis = basis.tabulate(0, reference)[0, 0, :, 0]
                dofs = V.dofmap.cell_dofs(int(cell))
                rows.extend([index] * len(dofs))
                columns.extend(dofs)
                entries.extend(local_basis)
                break
            else:
                raise ValueError(
                    f"Temperature sensor {index} at {point.tolist()} is outside the heat domain."
                )
        self.matrix = csr_matrix(
            (entries, (rows, columns)),
            shape=(len(points), V.dofmap.index_map.size_local),
            dtype=np.float64,
        )
        self.matrix.eliminate_zeros()
        self.matrix.sort_indices()

    def predict(self, T_dofs: np.ndarray) -> np.ndarray:
        r"""Evaluate the temperature at every physical sensor.

        Parameters
        ----------
        T_dofs : numpy.ndarray
            Temperatures in native V ordering, shape (N,).

        Returns
        -------
        numpy.ndarray
            Predicted temperatures in original sensor order, shape (S,).
        """
        values = _real_sensor_array(T_dofs, "Nodal temperature")
        if values.shape != (self.matrix.shape[1],):
            raise ValueError(
                f"Nodal temperature must have shape ({self.matrix.shape[1]},)."
            )
        return np.asarray(self.matrix @ values).reshape(-1)

    def transpose(self, weights: np.ndarray) -> np.ndarray:
        r"""Apply the dual sensor map, with no finite-element mass weighting.

        Parameters
        ----------
        weights : numpy.ndarray
            Sensor residual weights in measurement order, shape (S,).

        Returns
        -------
        numpy.ndarray
            Dual temperature weights in native V ordering, shape (N,).
        """
        weights = _sensor_vector(weights, self.matrix.shape[0], "Sensor weight")
        return np.asarray(self.matrix.T @ weights).reshape(-1)
