"""Frozen single-rank heat experiments and coordinate-labelled nodal fields."""

# type imports
from dataclasses import dataclass, field
from typing import Callable
import hashlib

# numerical imports
import numpy as np
from scipy.spatial import cKDTree

# mpi and dolfinx imports
from mpi4py import MPI
from dolfinx import fem, mesh
import basix
import basix.ufl
import ufl

# local imports
from domain_coefficient import HeatSource
from forward_solver import SteadyHeat2DForwardSolver


def _real_values(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real.")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _coordinate_order(coordinates):
    coordinates = _real_values(coordinates, "Field coordinates")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Field coordinates must have shape (N, 2).")
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    if len(order) > 1 and np.any(
        np.all(
            np.isclose(np.diff(coordinates[order], axis=0), 0.0, rtol=0.0, atol=1e-14),
            axis=1,
        )
    ):
        raise ValueError("Field coordinates are ambiguous: nodes must be unique.")
    return order


def _validate_space(V):
    if V.mesh.comm.size != 1:
        raise ValueError("A heat workspace requires a single MPI rank.")
    element = V.element
    if (
        V.mesh.topology.dim != 2
        or V.mesh.geometry.dim != 2
        or V.mesh.topology.cell_name() not in ("triangle", "quadrilateral")
        or element.value_shape.size != 0
        or element.basix_element.degree != 1
        or element.basix_element.family != basix.ElementFamily.P
        or element.basix_element.discontinuous
        or V.mesh.geometry.cmaps[0].degree != 1
    ):
        raise ValueError(
            "Heat experiments require scalar continuous affine P1/Q1 elements in 2D."
        )


def _physical_dof_coordinates(V):
    # For continuous P1/Q1, each local heat node is the corresponding
    # coordinate-element vertex. Associate through cells rather than sorting
    # tabulated interpolation coordinates: their floating-point push-forward
    # can perturb equal x coordinates and reorder entire physical columns.
    _validate_space(V)
    dofs = V.dofmap.list
    vertices = V.mesh.geometry.dofmaps[0]
    count = V.dofmap.index_map.size_local
    if dofs.shape != vertices.shape:
        raise ValueError(
            "Geometry and heat nodes must coincide for affine P1/Q1 elements."
        )
    association = np.full(count, -1, dtype=np.int64)
    association[dofs.ravel()] = vertices.ravel()
    if (
        np.any(association < 0)
        or not np.array_equal(association[dofs], vertices)
        or len(np.unique(association)) != count
    ):
        raise ValueError("Heat nodes must have a unique physical vertex association.")
    return V.mesh.geometry.x[association, :2].copy()


def canonical_dof_indices(V) -> np.ndarray:
    r"""Return native scalar P1/Q1 indices in lexicographic physical (x,y) order.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Scalar nodal heat space on one MPI rank.

    Returns
    -------
    numpy.ndarray
        Permutation selecting native values in canonical coordinate order.
    """
    _validate_space(V)
    return _coordinate_order(_physical_dof_coordinates(V))


@dataclass(frozen=True)
class HeatNumerics:
    r"""Fixed heat-solve tolerances, pending posterior accuracy qualification."""

    rtol: float = 1e-10
    atol: float = 1e-12
    max_it: int = 10000
    ksp_type: str = "cg"
    pc_type: str = "hypre"

    def __post_init__(self):
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if np.ndim(value) != 0 or np.iscomplexobj(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar.")
            if value < 0.0 or (name == "rtol" and value == 0.0):
                raise ValueError(
                    f"{name} must be {'positive' if name == 'rtol' else 'nonnegative'}."
                )
            object.__setattr__(self, name, value)
        if (
            isinstance(self.max_it, (bool, np.bool_))
            or not isinstance(self.max_it, (int, np.integer))
            or self.max_it <= 0
        ):
            raise ValueError("max_it must be a positive integer.")
        for name in ("ksp_type", "pc_type"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty PETSc solver name.")

    def petsc_options(self) -> dict:
        r"""Return fixed options enforcing failure on unconverged heat solves."""
        return {
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "ksp_rtol": self.rtol,
            "ksp_atol": self.atol,
            "ksp_max_it": self.max_it,
            "ksp_error_if_not_converged": True,
            "ksp_initial_guess_nonzero": False,
        }


@dataclass(frozen=True)
class NodalField:
    r"""Nodal field values paired with their physical two-dimensional coordinates."""

    values: np.ndarray
    coordinates: np.ndarray


FieldInput = float | Callable | fem.Function | NodalField


def _ordered_field(values, coordinates, model_coordinates):
    values = _real_values(values, "Field values")
    coordinates = _real_values(coordinates, "Field coordinates")
    _coordinate_order(coordinates)
    if values.shape != (len(coordinates),):
        raise ValueError("Field values must have shape (N,).")
    if coordinates.shape != model_coordinates.shape:
        raise ValueError(
            "Field coordinates must match the frozen mesh; cross-mesh transfer is not supported."
        )
    # Coordinate labels from a nodal push-forward can differ from the physical
    # vertices by roundoff. Match a unique node within 32 machine eps at the
    # coordinate scale. This only selects existing values; it never interpolates.
    tolerance = (
        32 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(model_coordinates)))
    )
    distance, nearest = cKDTree(coordinates).query(model_coordinates, k=2)
    order = nearest[:, 0]
    if np.any(distance[:, 0] > tolerance) or len(np.unique(order)) != len(order):
        raise ValueError(
            "Field coordinates must match the frozen mesh to floating-point roundoff; cross-mesh transfer is not supported."
        )
    if np.any(distance[:, 1] <= tolerance):
        raise ValueError(
            "Field coordinates are ambiguous: each heat node requires a unique label."
        )
    return values[order].copy()


def coerce_field(value: FieldInput, model: "FrozenHeatModel") -> np.ndarray:
    r"""Convert a real field to canonical nodal values, preserving negative log means.

    Parameters
    ----------
    value : float, callable, fem.Function or NodalField
        A scalar, existing UFL-style callable, or coordinate-labelled nodal field.
    model : FrozenHeatModel
        Physical mesh on which all nodal unknowns are retained.

    Returns
    -------
    numpy.ndarray
        Independent real finite field values in canonical coordinate order.
    """
    if isinstance(value, NodalField):
        return _ordered_field(value.values, value.coordinates, model.coordinates)
    if isinstance(value, fem.Function):
        _validate_space(value.function_space)
        return _ordered_field(
            value.x.array,
            _physical_dof_coordinates(value.function_space),
            model.coordinates,
        )
    if callable(value):
        domain = model._build_mesh()
        V = fem.functionspace(domain, ("Lagrange", 1))
        coefficient = HeatSource(value, domain, V).function
        if isinstance(coefficient, fem.Constant):
            return coerce_field(coefficient.value.item(), model)
        return coerce_field(coefficient, model)
    if np.isscalar(value):
        scalar = _real_values(value, "Field values")
        return np.full(len(model.coordinates), scalar.item(), dtype=np.float64)
    raise TypeError(
        "Nodal arrays require coordinates: supply NodalField(values, coordinates)."
    )


def _immutable_copy(values, dtype):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class FrozenHeatModel:
    r"""Physical heat experiment independent of the caller's mutable FEM state."""

    coordinates: np.ndarray
    cells: np.ndarray
    cell_type: str
    source: np.ndarray
    boundary_dofs: np.ndarray
    boundary_values: np.ndarray
    fingerprint: str = field(init=False)

    def __post_init__(self):
        for name in ("coordinates", "source", "boundary_values"):
            object.__setattr__(
                self,
                name,
                _immutable_copy(_real_values(getattr(self, name), name), np.float64),
            )
        for name in ("cells", "boundary_dofs"):
            object.__setattr__(
                self, name, _immutable_copy(getattr(self, name), np.int64)
            )
        digest = hashlib.sha256(self.cell_type.encode("ascii"))
        # Cell orientation is needed to rebuild the mesh; physical identity only
        # depends on which nodes form each cell, independent of local numbering.
        connectivity = np.sort(self.cells, axis=1)
        connectivity = connectivity[np.lexsort(connectivity.T[::-1])]
        for array in (
            self.coordinates,
            connectivity,
            self.source,
            self.boundary_dofs,
            self.boundary_values,
        ):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @classmethod
    def from_forward(cls, forward) -> "FrozenHeatModel":
        r"""Copy mesh, evaluated source, and actual boundary temperatures.

        Parameters
        ----------
        forward : SteadyHeat2DForwardSolver
            Caller-owned physical experiment. Its conductivity and temperature
            are neither modified nor included in the experiment fingerprint.

        Returns
        -------
        FrozenHeatModel
            Independent snapshot of the physical heat experiment.
        """
        if forward.mesh.comm.size != 1:
            raise ValueError("A frozen heat experiment requires a single MPI rank.")
        order = canonical_dof_indices(forward.V)
        coordinates = _physical_dof_coordinates(forward.V)[order]
        geometry = forward.mesh.geometry
        geometry_order = _coordinate_order(geometry.x[:, :2])
        if not np.array_equal(geometry.x[geometry_order, :2], coordinates):
            raise ValueError(
                "Geometry and heat nodes must coincide for affine P1/Q1 elements."
            )
        inverse = np.argsort(geometry_order)
        cells = inverse[geometry.dofmaps[0]].astype(np.int64)
        if forward.mesh.topology.cell_name() == "quadrilateral":
            vertices = coordinates[cells]
            if not np.allclose(
                vertices[:, 0] + vertices[:, 3],
                vertices[:, 1] + vertices[:, 2],
                rtol=0.0,
                atol=1e-14,
            ):
                raise ValueError(
                    "Q1 heat experiments require affine quadrilateral cells."
                )
        q = forward.q.function
        if isinstance(q, fem.Constant):
            source = np.full(len(order), _real_values(q.value, "Heat source").item())
        elif isinstance(q, fem.Function) and q.function_space == forward.V:
            source = _real_values(q.x.array, "Heat source")[order]
        else:
            raise ValueError(
                "Heat source must be scalar and defined on the heat space."
            )
        native_values = np.zeros(len(order), dtype=forward.T.x.array.dtype)
        imposed = {}
        for bc in forward.bcs:
            indices, _ = bc.dof_indices()
            values = np.zeros_like(native_values)
            bc.set(values)
            _real_values(values[indices], "Boundary temperatures")
            for index in indices:
                if int(index) in imposed and imposed[int(index)] != values[index]:
                    raise ValueError(
                        "Conflicting boundary temperatures at a heat node."
                    )
                imposed[int(index)] = values[index]
        inverse_dofs = np.argsort(order)
        boundary_dofs = np.array(sorted(inverse_dofs[list(imposed)]), dtype=np.int64)
        for index, value in imposed.items():
            native_values[index] = value
        boundary_values = native_values[order][boundary_dofs]
        return cls(
            coordinates,
            cells,
            forward.mesh.topology.cell_name(),
            source,
            boundary_dofs,
            boundary_values,
        )

    def _build_mesh(self):
        element = basix.ufl.element("Lagrange", self.cell_type, 1, shape=(2,))
        return mesh.create_mesh(
            MPI.COMM_SELF, self.cells.copy(), ufl.Mesh(element), self.coordinates.copy()
        )

    def build_forward(
        self, *, prefix: str, numerics: HeatNumerics
    ) -> SteadyHeat2DForwardSolver:
        r"""Build an independent full-field heat solve on MPI.COMM_SELF.

        Parameters
        ----------
        prefix : str
            Distinct PETSc option prefix for this workspace.
        numerics : HeatNumerics
            Fixed heat-solve settings.

        Returns
        -------
        SteadyHeat2DForwardSolver
            Workspace initialized with one conductivity unknown at every node.
        """
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("A nonempty distinct PETSc prefix is required.")
        domain = self._build_mesh()
        V = fem.functionspace(domain, ("Lagrange", 1))
        order = canonical_dof_indices(V)
        if not np.array_equal(_physical_dof_coordinates(V)[order], self.coordinates):
            raise ValueError(
                "Rebuilt mesh coordinates do not match the frozen experiment."
            )
        source = np.empty(len(order))
        source[order] = self.source
        boundary_dofs = order[self.boundary_dofs]
        return SteadyHeat2DForwardSolver.from_discrete_data(
            domain=domain,
            h_values=np.ones(len(order)),
            q_values=source,
            boundary_dofs=boundary_dofs,
            boundary_values=self.boundary_values,
            petsc_opts=numerics.petsc_options(),
            prefix=prefix,
        )
