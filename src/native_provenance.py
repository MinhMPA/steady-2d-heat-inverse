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


def _canonical_image(observed, prefix):
    """Name a proven sibling hardlink by its same-suffix active-prefix path."""
    if observed.is_relative_to(prefix) or not observed.is_relative_to(prefix.parent):
        return observed
    relative = observed.relative_to(prefix.parent)
    if len(relative.parts) < 2:
        return observed
    candidate = prefix.joinpath(*relative.parts[1:]).resolve()
    if not candidate.is_relative_to(prefix) or not candidate.is_file():
        return observed
    try:
        return candidate if observed.samefile(candidate) else observed
    except OSError:
        # An unprovable alias retains its observed identity and strict comparison.
        return observed


def _library_evidence(images, prefix, select, diagnostics=None):
    """Normalize before selection; verify observed bytes before deduplicating."""
    normalized = []
    for image in images:
        observed = Path(image).resolve()
        canonical = _canonical_image(observed, prefix)
        if observed.is_file() and select(canonical):
            normalized.append((canonical, observed, str(image)))
    by_path = {}
    by_basename = {}
    for canonical, observed, loader_path in sorted(normalized):
        digest = _sha(observed)
        if canonical != observed and (
            _sha(canonical) != digest or not observed.samefile(canonical)
        ):
            raise RuntimeError(
                "Native numerical-library alias changed during inspection: "
                f"{observed} -> {canonical}"
            )
        item = {"path": str(canonical), "sha256": digest}
        key = canonical.name.lower()
        if key in by_basename and by_basename[key]["path"] != item["path"]:
            raise RuntimeError(
                "Ambiguous numerical-library basename: "
                f"{key}: {by_basename[key]['path']} and {canonical}"
            )
        if canonical in by_path and by_path[canonical] != item:
            raise RuntimeError(
                "Native numerical-library bytes changed during inspection: "
                + str(canonical)
            )
        by_basename[key] = item
        by_path[canonical] = item
        if diagnostics is not None:
            diagnostics.append(
                {
                    "loader_path": loader_path,
                    "observed_path": str(observed),
                    "canonical_path": str(canonical),
                    "sha256": digest,
                    "sibling_samefile_verified": canonical != observed,
                }
            )
    return list(by_path.values())


def native_build_identity(*, image_diagnostics=None):
    """Return actual native evidence from this interpreter and fresh resolution.

    Fresh-process resolution uses the active interpreter, its search path and
    environment. Already-loaded parent numerical components must resolve to the
    same files; disagreement raises rather than concealing import-order effects.
    No evidence is cached: compiler output, package records and library bytes are
    inspected again on continuation. Missing optional native stacks are recorded.
    Executable/source searches affect identity through their resolved components;
    unused search entries and a shell CONDA_PREFIX label are not numerical inputs.
    Native loader controls remain explicit because they can change symbol loading.

    A sibling-installation path is replaced by its active-prefix same-suffix name
    only after samefile proof and byte verification. Schema 2 records this policy;
    old-source calculations require new qualification and cannot be migrated.
    Optional ``image_diagnostics`` receives fresh/parent loader observations and
    alias proofs separately from the equality-bearing return value. Loader paths
    are those supplied by _loaded_images (which already resolves symlinks).
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
    if image_diagnostics is not None:
        image_diagnostics.update({"fresh": [], "parent": []})
    libraries = _library_evidence(
        probe["images"],
        prefix,
        lambda path: (
            path.is_relative_to(prefix)
            or path.name.lower().startswith(_COMPONENT_PREFIXES)
            or "site-packages" in str(path)
        ),
        None if image_diagnostics is None else image_diagnostics["fresh"],
    )
    fresh = {Path(item["path"]).name.lower(): item for item in libraries}
    parents = _library_evidence(
        _loaded_images(),
        prefix,
        lambda path: (
            path.name.lower().startswith(_COMPONENT_PREFIXES)
            or path.name.lower() in fresh
        ),
        None if image_diagnostics is None else image_diagnostics["parent"],
    )
    for parent in parents:
        key = Path(parent["path"]).name.lower()
        if key not in fresh or parent != fresh[key]:
            raise RuntimeError(
                "Loaded parent and fresh-process numerical libraries disagree: "
                + parent["path"]
            )
    records = {}
    loaded_paths = {item["path"] for item in libraries}
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
        "schema": 2,
        "image_path_policy": "active_prefix_same_suffix_sibling_samefile_and_sha256",
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
