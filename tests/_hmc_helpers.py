"""Small physical heat experiments for the full-field posterior tests."""

import numpy as np
from dolfinx import fem
from forward_solver import SteadyHeat2DForwardSolver


def make_forward(nmesh=4, cell="triangle", h=1.0, q=1.0, bottom=300.0):
    return SteadyHeat2DForwardSolver(
        nmesh=nmesh, mesh_type=cell, h=h, q=q, DBC_value=bottom
    )


def state_arrays(fwd):
    h = fwd.h.function
    values = h.x.array if isinstance(h, fem.Function) else np.asarray(h.value)
    return values.copy(), fwd.T.x.array.copy()
