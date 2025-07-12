from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from petsc4py import PETSc
from dolfinx import mesh, fem
import ufl

UserInput    = Union[Union[int, float], fem.Constant, fem.Expression, Callable[[np.ndarray], np.ndarray]]

class BaseDomainCoefficient(ABC):
    """
    Base class for any coefficient defined on a DOLFINx mesh.
    Derived class *must* implement the `_build` method, which populates `self.function`.
    """

    def __init__(self, user_input: UserInput, mesh: mesh.Mesh, V: fem.FunctionSpace):
        self._user_input = user_input
        self._mesh = mesh
        self._V = V
        self.function = self._build()          # guaranteed by ABC

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
            return self._user_input
        if isinstance(self._user_input, (int, float)):
            return fem.Constant(self._mesh, PETSc.ScalarType(self._user_input))
        # Handle callable or fem.Expression inputs
        if isinstance(self._user_input, fem.Expression):
            ## If input is already a fem.Expression, interpolate directly onto the mesh
            f = fem.Function(self._V)
            f.interpolate(self._user_input)
            return f
        if callable(self._user_input):
            ## If input is python callable, pack the user-input callable into a fem.Expression before interpolation
            expr = fem.Expression(
                    self._user_input(ufl.SpatialCoordinate(self._mesh)),
                    self._V.element.interpolation_points()
                )
            f = fem.Function(self._V)
            f.interpolate(expr)
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

class ThermalConductivity(BaseDomainCoefficient):
    """
    Derived class for thermal conductivity coefficient h(x,y).
    """

    def _build(self):
        return self._coefficient_from_user_input()


class HeatSource(BaseDomainCoefficient):
    """
    Derived class for the heat source coefficient q(x,y).
    """

    def _build(self):
        return self._coefficient_from_user_input()
