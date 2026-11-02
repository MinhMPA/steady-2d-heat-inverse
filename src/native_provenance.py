"""Inspectable compiler and resolved native numerical-library provenance.

Inspection is called only for new calculations and continuation, never while
opening a read-only saved posterior. No optional sampling package is imported.
"""

from pathlib import Path
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig

# Resolve executable/source search paths to the components actually used.
# Unused PATH/PYTHONPATH entries and the shell CONDA_PREFIX label do not
# change a calculation; sys.prefix identifies the active numerical installation.
_LOADER_INPUTS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
)
_COMPONENT_PREFIXES = (
    "libpetsc",
    "libmpi",
    "libmpich",
    "libhypre",
    "libopenblas",
    "libdolfinx",
    "libbasix",
    "libblas",
    "liblapack",
    "libmumps",
    "libdmumps",
    "libmetis",
    "libptscotch",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_images():
    if sys.platform == "darwin":
        library = ctypes.CDLL(None)
        library._dyld_image_count.restype = ctypes.c_uint32
        library._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        library._dyld_get_image_name.restype = ctypes.c_char_p
        return sorted(
            {
                str(Path(library._dyld_get_image_name(i).decode()).resolve())
                for i in range(library._dyld_image_count())
            }
        )
    maps = Path("/proc/self/maps")
    if maps.exists():
        return sorted(
            {
                str(Path(line.split(maxsplit=5)[-1].strip()).resolve())
                for line in maps.read_text().splitlines()
                if len(line.split(maxsplit=5)) == 6
                and line.split(maxsplit=5)[-1].startswith("/")
            }
        )
    return []


def _compiler(name):
    declaration = os.environ.get(name) or sysconfig.get_config_var(name)
    command = shlex.split(declaration or ("cc" if name == "CC" else "c++"))
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "declaration": declaration,
            "status": "unavailable",
            "version": None,
            "executable_sha256": None,
        }
    resolved = str(Path(executable).resolve())
    result = subprocess.run(
        [resolved, *command[1:], "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "declaration": declaration,
        "command": command,
        "path": resolved,
        "version": (result.stdout + result.stderr).strip(),
        "returncode": result.returncode,
        "executable_sha256": _sha(resolved),
        "status": "inspected" if result.returncode == 0 else "failed",
    }


_PROBE = r"""
import sys, json, importlib
sys.path = json.loads(sys.stdin.read())
from native_provenance import _loaded_images
imports = {}
for name in ('numpy', 'scipy.linalg', 'scipy.sparse.linalg', 'scipy.optimize', 'dolfinx', 'petsc4py.PETSc', 'mpi4py.MPI'):
    try:
        module = importlib.import_module(name)
        imports[name] = {'status':'loaded', 'path':getattr(module, '__file__', None)}
    except (ImportError, OSError) as exc:
        imports[name] = {'status':'unavailable', 'error':str(exc)}
print('S2DHI_NATIVE_JSON=' + json.dumps({'imports':imports, 'images':_loaded_images()}))
"""


def _sampling_artifacts():
    """Hash installed JAXlib native files without loading JAX or JAXlib."""
    try:
        distribution = importlib.metadata.distribution("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return {"jaxlib": {"status": "absent", "version": None, "libraries": []}}
    paths = sorted(
        {
            Path(distribution.locate_file(name)).resolve()
            for name in distribution.files or []
            if str(name).endswith((".so", ".dylib", ".dll", ".pyd"))
        }
    )
    libraries = [{"path": str(path), "sha256": _sha(path)} for path in paths]
    return {
        "jaxlib": {
            "status": "installed_not_loaded",
            "version": distribution.version,
            "libraries": libraries,
        }
    }


def native_build_identity():
    """Return actual native evidence from this interpreter and fresh resolution.

    Fresh-process resolution uses the active interpreter, its search path and
    environment. Already-loaded parent numerical components must resolve to the
    same files; disagreement raises rather than concealing import-order effects.
    No evidence is cached: compiler output, package records and library bytes are
    inspected again on continuation. Missing optional native stacks are recorded.
    Executable/source searches affect identity through their resolved components;
    unused search entries and a shell CONDA_PREFIX label are not numerical inputs.
    Native loader controls remain explicit because they can change symbol loading.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        input=json.dumps(sys.path),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("S2DHI_NATIVE_JSON=")
    ]
    if result.returncode or len(lines) != 1:
        raise RuntimeError(
            "Native numerical-library inspection failed: " + result.stderr[-2000:]
        )
    probe = json.loads(lines[0].split("=", 1)[1])
    prefix = Path(sys.prefix).resolve()
    selected = [
        Path(path)
        for path in probe["images"]
        if Path(path).is_file()
        and (
            Path(path).is_relative_to(prefix)
            or Path(path).name.lower().startswith(_COMPONENT_PREFIXES)
            or "site-packages" in str(path)
        )
    ]
    libraries = [{"path": str(path), "sha256": _sha(path)} for path in selected]
    fresh = {Path(item["path"]).name.lower(): item for item in libraries}
    for parent in _loaded_images():
        path = Path(parent)
        key = path.name.lower()
        numerical = key.startswith(_COMPONENT_PREFIXES) or key in fresh
        if not numerical or not path.is_file():
            continue
        if (
            key not in fresh
            or str(path) != fresh[key]["path"]
            or _sha(path) != fresh[key]["sha256"]
        ):
            raise RuntimeError(
                "Loaded parent and fresh-process numerical libraries disagree: "
                + str(path)
            )
    records = {}
    loaded_paths = {str(path) for path in selected}
    important = {
        "petsc",
        "mpich",
        "openmpi",
        "hypre",
        "libopenblas",
        "fenics-libdolfinx",
        "fenics-libbasix",
    }
    for entry in sorted((prefix / "conda-meta").glob("*.json")):
        info = json.loads(entry.read_text())
        owned = [str((prefix / name).resolve()) for name in info.get("files", [])]
        if info.get("name") in important or loaded_paths.intersection(owned):
            records[info["name"]] = {
                key: info.get(key)
                for key in (
                    "version",
                    "build",
                    "build_number",
                    "sha256",
                    "channel",
                    "subdir",
                )
            }
            records[info["name"]]["record_sha256"] = _sha(entry)
    return {
        "schema": 1,
        "compiler": {name: _compiler(name) for name in ("CC", "CXX")},
        "interpreter": str(Path(sys.executable).resolve()),
        "prefix": str(prefix),
        "platform": platform.system(),
        "loader_inputs": {name: os.environ.get(name) for name in _LOADER_INPUTS},
        "resolution": "fresh_process_verified_against_loaded_parent_components",
        "imports": probe["imports"],
        "libraries": libraries,
        "conda_builds": records,
        "optional_sampling_artifacts": _sampling_artifacts(),
    }
