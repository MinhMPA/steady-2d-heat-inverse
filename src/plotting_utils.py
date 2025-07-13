import numpy as np
from mpi4py import MPI
import pyvista as pv
from dolfinx.plot import vtk_mesh
from typing import Sequence
from numpy.typing import ArrayLike

def plot_scalar_mesh(mesh, data: ArrayLike, name: str,
                 cmap: str = "viridis", show_edges: bool = False,
                 n_labels: int = 5,
                 user_scalar_bar: dict | None = None,
                 return_plotter: bool = False,
                ):
    """
    Plot a scalar field on a Dolfinx mesh using PyVista.

    Parameters
    ----------
    mesh: dolfinx.mesh
    data: ArrayLike, (n_cells,)
        A scalar field on the mesh.
    name: str
        Displayed name of the scalar field.
    cmap: str, optional. Default is "viridis".
        Colormap.
    show_edges: bool, optional. Default is False.
        Whether to show mesh edges.
    n_labels: int, optional. Default is 5.
        Number of tick labels on the color bar.
    user_scalar_bar: dict, optional. Default is None.
        User-defined additional arguments for the scalar bar.
    """
    if MPI.COMM_WORLD.rank != 0:
        return

    cell_topology, cell_type, cell_geometry = vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(cell_topology, cell_type, cell_geometry)
    grid.point_data[name] = np.asarray(data, dtype=np.float64)

    # Automatically determine the format and the number of significant digits for the color bar
    dr = np.max(data) - np.min(data)
    digits = max(0, int(np.ceil(-np.log10(dr)))) if dr > 0 else 0
    if digits > 3:
        fmt = f"%.{digits}e"
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
    pl.add_mesh(grid, scalars=name, cmap=cmap, show_edges=show_edges,
                scalar_bar_args=scalar_bar)
    pl.show()
    if return_plotter:
        return grid, pl
    else:
        return grid
