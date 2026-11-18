"""Full nodal Gaussian priors for dimensionless log conductivity."""

# type imports
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
import copy
import hashlib
import json

# numerical imports
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

# dolfinx imports
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import ufl

# local imports
from heat_model import (
    FrozenHeatModel,
    NodalField,
    _immutable_copy,
    _physical_dof_coordinates,
    _real_values,
    canonical_dof_indices,
    coerce_field,
)


@runtime_checkable
class ConductivityPrior(Protocol):
    r"""A bound prior declaring all inferred coordinates and their probability law.

    Coordinate dimension belongs to the prior and need not equal the number of
    conductivity nodes. Inverse initialization is a separate capability.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def coordinate_schema(self) -> dict[str, tuple[int, ...]]: ...

    @property
    def coordinate_law(self) -> str: ...

    def to_log_h(self, z: np.ndarray) -> np.ndarray: ...

    def to_h(self, z: np.ndarray) -> np.ndarray: ...

    def pullback(self, z: np.ndarray, grad_log_h: np.ndarray) -> np.ndarray: ...

    def energy(self, z: np.ndarray) -> float: ...

    def grad_energy(self, z: np.ndarray) -> np.ndarray: ...

    def record(self) -> dict: ...


@runtime_checkable
class InvertibleConductivityPrior(ConductivityPrior, Protocol):
    r"""A bound prior that can initialize its complete state from conductivity."""

    def from_h(self, h: np.ndarray) -> np.ndarray: ...


def _positive_scalar(value, name):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    value = _real_values(value, name).item()
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _ordered_cells(cells):
    cells = np.sort(cells, axis=1)
    return cells[np.lexsort(cells.T[::-1])]


def _binding_order(model, V):
    order = canonical_dof_indices(V)
    if (
        V.mesh.topology.cell_name() != model.cell_type
        or not np.array_equal(_physical_dof_coordinates(V)[order], model.coordinates)
        or not np.array_equal(
            _ordered_cells(np.argsort(order)[V.dofmap.list]),
            _ordered_cells(model.cells),
        )
    ):
        raise ValueError(
            "Prior space must match the frozen mesh coordinates and cells."
        )
    return order


def _assemble_sparse(form):
    matrix = assemble_matrix(fem.form(form))
    try:
        matrix.assemble()
        indptr, indices, values = matrix.getValuesCSR()
        return csr_matrix(
            (values.copy(), indices.copy(), indptr.copy()), shape=matrix.getSize()
        )
    finally:
        matrix.destroy()


@dataclass(frozen=True, kw_only=True)
class GaussianLogConductivity:
    r"""Specify a proper Gaussian field for :math:`m=\log(h/h_{ref})`.

    Parameters
    ----------
    mu : float, callable, NodalField or dolfinx.fem.Function
        Mean dimensionless log conductivity, evaluated and copied when bound.
    kappa : float
        Positive inverse correlation scale in the physical coordinate units.
    tau : float
        Positive amplitude precision in the elliptic prior construction.
    h_ref : float
        Positive reference conductivity, in the same units as the inferred field.
    """

    mu: float | Callable | NodalField | fem.Function
    kappa: float
    tau: float
    h_ref: float

    def __post_init__(self):
        for name in ("kappa", "tau", "h_ref"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))

    def bind(self, model: FrozenHeatModel, V) -> "BoundGaussianPrior":
        r"""Evaluate the mean and bind all conductivity nodes to a physical mesh.

        Parameters
        ----------
        model : FrozenHeatModel
            Frozen physical mesh defining canonical nodal order.
        V : dolfinx.fem.FunctionSpace
            Scalar P1/Q1 space on that same mesh, on one MPI rank.

        Returns
        -------
        BoundGaussianPrior
            Independent prior with natural Neumann conditions on every wall.
        """
        return BoundGaussianPrior(self, model, V)


class BoundGaussianPrior:
    r"""Sparse full-field map :math:`m=\mu+\tau^{-1}A^{-1}M_L^{1/2}z`.

    Here :math:`A=\kappa^2M+K`, and :math:`M_L=\mathrm{diag}(M\mathbf{1})`.
    Neither matrix imposes the temperature Dirichlet condition. All public
    vectors use canonical physical node order; matrices retain native FEM order.

    Parameters
    ----------
    specification : GaussianLogConductivity
        Positive physical prior parameters and mean field.
    model : FrozenHeatModel
        Frozen physical mesh defining canonical nodal order.
    V : dolfinx.fem.FunctionSpace
        Matching scalar P1/Q1 space on a single MPI rank.
    """

    def __init__(self, specification: GaussianLogConductivity, model, V):
        order = _binding_order(model, V)
        self._order = _immutable_copy(order, np.int64)
        self._mu = _immutable_copy(coerce_field(specification.mu, model), np.float64)
        self._dimension = len(self._mu)
        self._tau = specification.tau
        self._h_ref = specification.h_ref
        trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
        mass = _assemble_sparse(trial * test * ufl.dx)
        stiffness = _assemble_sparse(
            ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        )
        lumped = np.asarray(mass @ np.ones(self.dimension)).ravel()
        if not np.isfinite(lumped).all() or np.any(lumped <= 0.0):
            raise ValueError("The prior requires positive finite lumped nodal masses.")
        self._sqrt_lumped = np.sqrt(lumped)
        self._A = (specification.kappa**2 * mass + stiffness).tocsc()
        self._factor = splu(self._A)
        mesh_digest = hashlib.sha256(model.cell_type.encode("ascii"))
        for array in (model.coordinates, _ordered_cells(model.cells)):
            mesh_digest.update(str(array.shape).encode("ascii"))
            mesh_digest.update(array.tobytes())
        self._record = {
            "family": "gaussian_log_conductivity",
            "dimension": self.dimension,
            "coordinate_schema": {"z": [self.dimension]},
            "coordinate_law": self.coordinate_law,
            "boundary_condition": "natural_neumann",
            "mesh_fingerprint": mesh_digest.hexdigest(),
            "mu": self._mu.tolist(),
            "kappa": specification.kappa,
            "tau": self._tau,
            "h_ref": self._h_ref,
        }
        self._record["fingerprint"] = hashlib.sha256(
            json.dumps(self._record, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()

    @property
    def dimension(self) -> int:
        r"""Number of independent Gaussian coordinates; every node is retained."""
        return self._dimension

    @property
    def coordinate_schema(self) -> dict[str, tuple[int, ...]]:
        r"""Complete inferred coordinates for this Gaussian field."""
        return {"z": (self.dimension,)}

    @property
    def coordinate_law(self) -> str:
        r"""Probability law of the independent prior coordinates."""
        return "standard_normal"

    def _vector(self, values, name):
        values = _real_values(values, name)
        if values.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},).")
        return values

    def _native(self, values):
        native = np.empty(self.dimension, dtype=np.float64)
        native[self._order] = values
        return native

    def to_log_h(self, z: np.ndarray) -> np.ndarray:
        r"""Map Gaussian coordinates to dimensionless nodal log conductivity.

        Parameters
        ----------
        z : numpy.ndarray
            Independent standard-normal coordinates, shape (dimension,).

        Returns
        -------
        numpy.ndarray
            Values of log(h/h_ref), in canonical node order.
        """
        z = self._vector(z, "Prior coordinates")
        fluctuation = self._factor.solve(self._sqrt_lumped * self._native(z))
        return self._mu + fluctuation[self._order] / self._tau

    def to_h(self, z: np.ndarray) -> np.ndarray:
        r"""Transform Gaussian coordinates to positive physical conductivity.

        Parameters
        ----------
        z : numpy.ndarray
            Independent standard-normal coordinates, shape (dimension,).

        Returns
        -------
        numpy.ndarray
            Conductivity in canonical node order. Exponential overflow remains
            nonfinite so the posterior can classify the proposal explicitly.
        """
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.exp(np.log(self._h_ref) + self.to_log_h(z))

    def from_h(self, h: np.ndarray) -> np.ndarray:
        r"""Recover Gaussian coordinates from positive finite conductivity.

        Parameters
        ----------
        h : numpy.ndarray
            Positive physical conductivity in canonical node order.

        Returns
        -------
        numpy.ndarray
            Coordinates tau M_L^(-1/2) A (log(h/h_ref) - mu).
        """
        h = self._vector(h, "Initial conductivity")
        if np.any(h <= 0.0):
            raise ValueError("Initial conductivity must be positive.")
        # Separate logarithms avoid underflow/overflow in the physical ratio.
        fluctuation = self._native(np.log(h) - np.log(self._h_ref) - self._mu)
        native_z = self._tau * (self._A @ fluctuation) / self._sqrt_lumped
        return native_z[self._order]

    def pullback(self, z: np.ndarray, grad_log_h: np.ndarray) -> np.ndarray:
        r"""Transform a log-field derivative by the exact transpose prior map.

        Parameters
        ----------
        z : numpy.ndarray
            Current Gaussian coordinates, shape (dimension,).
        grad_log_h : numpy.ndarray
            Nodal derivative with respect to log(h/h_ref), in canonical order.

        Returns
        -------
        numpy.ndarray
            Derivative with respect to Gaussian coordinates.
        """
        self._vector(z, "Prior coordinates")
        gradient = self._native(self._vector(grad_log_h, "Log-conductivity gradient"))
        native_gradient = self._sqrt_lumped * self._factor.solve(gradient, trans="T")
        return native_gradient[self._order] / self._tau

    def energy(self, z: np.ndarray) -> float:
        r"""Evaluate the standard-normal energy, with no conductivity Jacobian.

        Parameters
        ----------
        z : numpy.ndarray
            Independent Gaussian coordinates, shape (dimension,).

        Returns
        -------
        float
            One half of the squared Euclidean coordinate norm.
        """
        z = self._vector(z, "Prior coordinates")
        return float(0.5 * (z @ z))

    def grad_energy(self, z: np.ndarray) -> np.ndarray:
        r"""Return the standard-normal energy derivative.

        Parameters
        ----------
        z : numpy.ndarray
            Independent Gaussian coordinates, shape (dimension,).

        Returns
        -------
        numpy.ndarray
            Independent copy of z.
        """
        return self._vector(z, "Prior coordinates").copy()

    def record(self) -> dict:
        r"""Return the evaluated prior physics, independent of initial states.

        Returns
        -------
        dict
            Independent JSON-compatible parameters, evaluated mean, mesh
            identity, coordinate law, and their deterministic fingerprint.
        """
        return copy.deepcopy(self._record)
