# type imports
from abc import ABC, abstractmethod
from typing import Literal, Callable, Union

# container imports
import pandas as pd

# numerical imports
import numpy as np
from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator

# pde imports
from petsc4py import PETSc
import ufl

# dolphinx imports
from dolfinx import mesh, fem

# local imports
from plotting_utils import plot_scalar_mesh

ScalarLike = Union[int, float]
CallableLike = Callable[[np.ndarray], np.ndarray]
TableLike = Union[np.ndarray, pd.DataFrame]
UserInput = Union[ScalarLike, fem.Constant, fem.Expression, CallableLike, TableLike]


class BaseDomainCoefficient(ABC):
    """
    Base class for any coefficient defined on a DOLFINx mesh.
    Derived class *must* implement the `_build` method, which populates `self.function`.
    """

    def __init__(
        self,
        user_input: UserInput,
        mesh: mesh.Mesh,
        V: fem.FunctionSpace,
        *,
        tab_interpolator: Literal["rbf", "ct"] = "ct",
    ):
        self._user_input = user_input
        self._mesh = mesh
        self._V = V
        self.function = self._build()
        self._tab_interpolator = tab_interpolator

    def _coefficient_from_user_input(self) -> Union[fem.Constant, fem.Function]:
        """
        Map user input into an object that UFL can treat as a coefficient.

        Parameters
        ----------
        Either of the following types are supported:
            - scalar (int, float) or fem.Constant
            - callable or fem.Expression

        Returns
        -------
        The coefficient defined on the domain. Either of the following types, depending on input:
            - fem.Constant for scalar and fem.Constant inputs
            - fem.Function for callable or fem.Expression inputs, interpolated onto the mesh
        """
        # Handle scalar (int,float) or fem.Constant inputs
        if isinstance(self._user_input, fem.Constant):
            self.constant = True
            return self._user_input
        if isinstance(self._user_input, (int, float)):
            self.constant = True
            return fem.Constant(self._mesh, PETSc.ScalarType(self._user_input))
        # Handle callable or fem.Expression inputs
        if isinstance(self._user_input, fem.Expression):
            self.constant = False
            ## If input is already a fem.Expression, interpolate directly onto the mesh
            f = fem.Function(self._V)
            f.interpolate(self._user_input)
            return f
        if callable(self._user_input):
            domain_coeff = self._user_input(ufl.SpatialCoordinate(self._mesh))
            ## Edge case: If the user-input callable actually returns a constant coefficient, convert it to fem.Constant
            if isinstance(domain_coeff, (int, float, PETSc.ScalarType)):
                self.constant = True
                return fem.Constant(self._mesh, PETSc.ScalarType(domain_coeff))
            ## If the user-input callable really returns a spatially-varying coefficient, pack it into a fem.Expression before interpolation
            else:
                self.constant = False
                expr = fem.Expression(
                    domain_coeff, self._V.element.interpolation_points()
                )
                f = fem.Function(self._V)
                f.interpolate(expr)
                return f
        if isinstance(self._user_input, (np.ndarray, pd.DataFrame)):
            self.constant = False
            pts, vals = self._parse_tab(self._user_input)
            ## piece-wise cubic interpolation, rescale to unit square before interpolating
            ## no extrapolation, values outside of the point convex hull can be specified with `fill_value`
            ## only works in 2D, but memory ~ O(N log N)
            # interp = CloughTocher2DInterpolator(pts, vals, fill_value=vals.mean(), rescale=True)
            ## radial basis function interpolation, cubic kernel with degree-1 polynomial added
            ## more hyperparameters but offers smoothing at sampled data points and extrapolation
            ## works in any dimension, but memory ~ O(N^2), can adjust neighbors to reduce memory usage
            interp = RBFInterpolator(
                pts, vals, kernel="cubic", neighbors=None, smoothing=0.0, degree=1
            )
            f = fem.Function(self._V)

            def interpolate_func(x):
                points = np.column_stack([x[0], x[1]])
                values = interp(points)
                return values

            f.interpolate(interpolate_func)
            return f

        raise TypeError(
            f"Unsupported coefficient parameter of type "
            f"{type(self._user_input).__name__}"
        )

    @abstractmethod
    def _build(self):
        """
        User must implement their own `_build()` method when deriving from the base class.
        """

        ...

    def _parse_tab(self, tab: TableLike):
        """
        Parse tabulated input `np.ndarray` or `pd.DataFrame`.

        Parameters
        ----------
        Either of the following types are supported:
            - `np.ndarray` ([x,y],value).
            - `pd.DataFrame` (x|y|value), case-insensitive.

        Returns
        -------
        pts (N,2) ndarray of points
        vals (N,) ndarray of values
        """
        if isinstance(tab, pd.DataFrame):
            cols = [c.lower() for c in tab.columns]
            try:
                pts = np.column_stack(
                    [tab[cols.index("x")].values, tab[cols.index("y")].values]
                )
                vals = tab[cols.index("value")].values
            except ValueError:
                raise ValueError("pd.DataFrame format must be (x|y|value).")
        elif isinstance(tab, np.ndarray):
            if tab.shape[-1] != 3:
                raise ValueError("np.ndarray format must be (N,[x,y,value]).")
            pts = tab[:, :2]
            vals = tab[:, -1]
        else:
            raise TypeError(
                f"Unsupported tabulated input type: {type(tab).__name__}. "
                "Expected np.ndarray or pd.DataFrame."
            )
        return pts, vals


class ThermalConductivity(BaseDomainCoefficient):
    """
    Derived class for thermal conductivity coefficient h(x,y).
    """

    def _build(self):
        return self._coefficient_from_user_input()

    def plot_input_thermal_conductivity(self, **kwargs):
        """
        Plot the input thermal conductivity coefficient on a pyvista.UnstructuredGrid.
        """
        if self.constant:
            vals = self.function.value
            kwargs.update(user_scalar_bar={"n_labels": 3})
        else:
            vals = self.function.x.array[: self._mesh.geometry.x.shape[0]]
        grid_plot = plot_scalar_mesh(
            self._mesh, vals, "h(x,y)", cmap="plasma", **kwargs
        )
        return grid_plot


class HeatSource(BaseDomainCoefficient):
    """
    Derived class for the heat source coefficient q(x,y).
    """

    def _build(self):
        return self._coefficient_from_user_input()

    def plot_input_heat_source(self, **kwargs):
        """
        Plot the input heat source coefficient on a pyvista.UnstructuredGrid.
        """
        if self.constant:
            vals = self.function.value
            kwargs.update(user_scalar_bar={"n_labels": 3})
        else:
            vals = self.function.x.array[: self._mesh.geometry.x.shape[0]]
        grid_plot = plot_scalar_mesh(
            self._mesh, vals, "q(x,y)", cmap="plasma", **kwargs
        )
        return grid_plot
