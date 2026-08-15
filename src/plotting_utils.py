# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# dolfinx imports
from dolfinx.plot import vtk_mesh

# type imports
from typing import Any
from numpy.typing import ArrayLike


def plot_scalar_mesh(
    mesh,
    data: ArrayLike,
    name: str,
    cmap: str = "viridis",
    show_edges: bool = False,
    n_labels: int = 5,
    user_scalar_bar: dict | None = None,
    return_plotter: bool = False,
    **mesh_kwargs: Any,
):
    """
    Plot a scalar field on a Dolfinx mesh using PyVista.

    Parameters
    ----------
    mesh : dolfinx.mesh.
    data : (n_cells,) scalar field on the mesh.
    name : displayed name of the scalar field.
    cmap : colormap.
    show_edges: whether to show mesh edges.
    n_labels: number of tick labels on the color bar.
    user_scalar_bar: user-defined additional arguments for the scalar bar.
    """
    # Imported lazily so that compute-only paths stay importable without VTK.
    # `pyvista` is an optional extra: `pip install -e ".[plot]"`.
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise ImportError(
            "Plotting requires the optional 'plot' extra, which is deliberately "
            "absent from the compute environment because pyvista/vtk conflict with "
            'fenics-dolfinx 0.11. Install it with: pip install -e ".[plot]"'
        ) from exc

    if MPI.COMM_WORLD.rank != 0:
        return

    cell_topology, cell_type, cell_geometry = vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(cell_topology, cell_type, cell_geometry)
    grid.point_data[name] = np.asarray(data, dtype=np.float64)

    # Automatically determine the format and the number of significant digits for the color bar
    dr = np.max(data) - np.min(data)
    digits = max(0, int(np.ceil(-np.log10(dr)))) if dr > 0 else 0
    if digits > 3:
        fmt = f"%.1e"
    else:
        fmt = f"%.{digits}f"

    # Define default colorbar arguments, update arguments if provided
    scalar_bar = {
        "title": name,
        "fmt": fmt,
        "n_labels": n_labels,
        "font_family": "arial",
        "title_font_size": 20,
        "label_font_size": 14,
    }
    if user_scalar_bar:
        scalar_bar.update(user_scalar_bar)

    pl = pv.Plotter()
    pl.add_mesh(
        grid,
        scalars=name,
        cmap=cmap,
        show_edges=show_edges,
        scalar_bar_args=scalar_bar,
        **mesh_kwargs,
    )
    pl.show()
    if return_plotter:
        return grid, pl
    else:
        return grid
