# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The sources are installed as *flat top-level modules* (see pyproject.toml
# [tool.setuptools] py-modules), so `src/` itself goes on the path -- not its parent.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "steady-2d-heat-inverse"
author = "Minh Nguyen"
copyright = "2026, Minh Nguyen"
release = "0.1"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
# `superpowers/` holds implementation plans, not published pages -- keep it out of the
# toctree so `-W` does not fail on "document isn't included in any toctree".
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers"]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

# -- autodoc -----------------------------------------------------------------

# The scientific stack is conda-only (fenics-dolfinx, petsc4py, ...) and far too
# heavy for a Read the Docs builder. Mock it so autodoc can import the modules and
# read their docstrings without any of it installed. Submodules such as
# `dolfinx.fem.petsc` and `numpy.typing` are mocked automatically.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "mpi4py",
    "petsc4py",
    "ufl",
    "basix",
    "dolfinx",
    "pyvista",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_member_order = "bysource"
autoclass_content = "both"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
