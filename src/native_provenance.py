"""Inspectable compiler and resolved native numerical-library provenance.

Inspection is called only for new calculations and continuation, never while
opening a read-only saved posterior. No optional sampling package is imported.
"""

from pathlib import Path
import ctypes
import hashlib
import importlib.metadata
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType

# Resolve executable/source search paths to the components actually used.
# Unused PATH/PYTHONPATH entries and the shell CONDA_PREFIX label do not
# change a calculation; sys.prefix identifies the active numerical installation.
_LOADER_INPUTS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_FORCE_FLAT_NAMESPACE",
)
_COMPONENT_PREFIXES = (
    "libpetsc",
    "libslepc",
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
from native_provenance import _loaded_images, _package_context
imports = {}
for name in ('numpy', 'scipy.linalg', 'scipy.sparse.linalg', 'scipy.optimize', 'scipy.interpolate', 'dolfinx', 'petsc4py.PETSc', 'mpi4py.MPI'):
    try:
        module = importlib.import_module(name)
        imports[name] = {'status':'loaded', 'path':getattr(module, '__file__', None)}
    except (ImportError, OSError) as exc:
        imports[name] = {'status':'unavailable', 'error':str(exc)}
images = _loaded_images()
print('S2DHI_NATIVE_JSON=' + json.dumps({'imports':imports, 'images':images, 'packages':_package_context(images, sys.prefix)}))
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


def _package_context(images, prefix):
    """Prove imported package origins and file ownership without importing them.

    Only stored module dictionaries and distribution file lists are inspected;
    lazy module accessors and package dependency traversal are deliberately absent.
    """
    prefix = Path(prefix).resolve()
    paths = {_canonical_image(Path(path).resolve(), prefix) for path in images}
    roots, modules, verified = {}, {}, {}
    for name, module in tuple(sys.modules.items()):
        # CFFI namespaces can report __class__=ModuleType without being modules.
        # The base descriptor below requires the object's actual concrete type.
        if not issubclass(type(module), ModuleType):
            continue
        data = ModuleType.__getattribute__(module, "__dict__")
        filename = data.get("__file__")
        if not isinstance(filename, str):
            continue
        spec = data.get("__spec__")
        attributes = {} if spec is None else object.__getattribute__(spec, "__dict__")
        origin = attributes.get("origin")
        declared = data.get("__name__", name)
        spec_name = attributes.get("name")
        path = _canonical_image(Path(filename).resolve(), prefix)
        consistent = (
            isinstance(origin, str)
            and _canonical_image(Path(origin).resolve(), prefix) == path
            and isinstance(spec_name, str)
            and sys.modules.get(spec_name) is module
        )
        if "." not in name and consistent and name == declared:
            search = data.get("__path__")
            if isinstance(search, (list, tuple)) and len(search) == 1:
                if Path(search[0]).resolve() == Path(filename).resolve().parent:
                    roots[name] = {"root": str(path.parent), "origin": str(path)}
        if filename.endswith(tuple(EXTENSION_SUFFIXES)):
            # sys.modules may also index the same extension by a short alias;
            # its own declared/spec names identify the actual Python origin.
            names = (
                {spec_name}
                if consistent
                else {
                    value for value in (declared, spec_name) if isinstance(value, str)
                }
            )
            modules.setdefault(path, set()).update(names)
            paths.add(path)
            if consistent:
                verified.setdefault(path, set()).add(spec_name)
    # Limit ownership inspection to imported packages containing observed files.
    roots = {
        name: info
        for name, info in roots.items()
        if any(path.is_relative_to(Path(info["root"])) for path in paths)
    }
    wanted = paths | {Path(info["origin"]) for info in roots.values()}
    owners = {path: set() for path in wanted}
    sites = {str(Path(info["root"]).parent) for info in roots.values()}
    targets_by_base = {}
    for distribution in importlib.metadata.distributions(path=sorted(sites)):
        files = distribution.files or []
        metadata = [name for name in files if str(name).endswith(".dist-info/METADATA")]
        # Some RECORD files omit METADATA; the distribution's declared path is
        # still obtained from its metadata reader, not inferred from a basename.
        location = getattr(distribution, "_path", None)
        if metadata:
            owner = str(Path(distribution.locate_file(metadata[0])).resolve())
        elif isinstance(location, Path) and (location / "METADATA").is_file():
            owner = str((location / "METADATA").resolve())
        else:
            continue
        base = Path(distribution.locate_file("")).resolve()
        if base not in targets_by_base:
            targets_by_base[base] = {
                str(path.relative_to(base)): path
                for path in wanted
                if path.is_relative_to(base)
            }
        targets = targets_by_base[base]
        for filename in files:
            if str(filename) in targets:
                actual = Path(distribution.locate_file(filename)).resolve()
                if actual in owners:
                    owners[actual].add(owner)
    packages = {}
    for name, info in roots.items():
        owned = owners[Path(info["origin"])]
        if len(owned) == 1:
            packages[name] = {**info, "owner": next(iter(owned))}
    evidence = {}
    for path in sorted(paths):
        names = sorted(modules.get(path, ()))
        candidates = [
            name
            for name, info in packages.items()
            if path.is_relative_to(Path(info["root"]))
            and owners[path] == {info["owner"]}
        ]
        claims = {name.split(".")[0] for name in names}
        valid = len(candidates) == 1 and (
            not names or (claims == {candidates[0]} and path in verified)
        )
        if valid:
            package = candidates[0]
            evidence[str(path)] = {
                "package": package,
                "relative": str(path.relative_to(Path(packages[package]["root"]))),
                "modules": names,
                "origins": sorted(verified.get(path, ())),
                "valid": True,
            }
        elif names or any(
            path.is_relative_to(Path(info["root"])) for info in roots.values()
        ):
            evidence[str(path)] = {"modules": names, "valid": False}
    return {"packages": packages, "files": evidence}


def _component_key(path, context):
    """Core roles precede package-qualified extension/helper identities."""
    entry = context["files"].get(str(path), {})
    if not path.name.lower().startswith(_COMPONENT_PREFIXES) and entry.get("valid"):
        return (
            "package",
            entry["package"],
            entry["relative"],
            tuple(entry.get("origins", entry.get("modules", ()))),
        )
    return ("runtime", path.name.lower())


def _library_evidence(images, prefix, select, diagnostics=None, *, component_key=None):
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
        key = (
            canonical.name.lower()
            if component_key is None
            else component_key(canonical)
        )
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
    fresh_context = probe.get("packages", {"packages": {}, "files": {}})
    parent_images = _loaded_images()
    parent_context = _package_context(parent_images, prefix)
    if image_diagnostics is not None:
        image_diagnostics.update({"fresh": [], "parent": [], "excluded_parent": []})
    libraries = _library_evidence(
        probe["images"],
        prefix,
        lambda path: (
            path.is_relative_to(prefix)
            or path.name.lower().startswith(_COMPONENT_PREFIXES)
            or "site-packages" in str(path)
        ),
        None if image_diagnostics is None else image_diagnostics["fresh"],
        component_key=lambda path: _component_key(path, fresh_context),
    )
    fresh = {
        _component_key(Path(item["path"]), fresh_context): item for item in libraries
    }
    fresh_names = {Path(item["path"]).name.lower() for item in libraries}
    numerical_packages = set(fresh_context["packages"]) | {
        name.split(".")[0] for name in probe["imports"]
    }
    fresh_paths = {item["path"] for item in libraries}
    for path, entry in fresh_context["files"].items():
        claims = {name.split(".")[0] for name in entry.get("modules", [])}
        if claims & numerical_packages and (
            not entry.get("valid") or path not in fresh_paths
        ):
            raise RuntimeError(
                "Missing or ambiguous fresh numerical module origin: " + path
            )

    def numerical_origin(path):
        entry = parent_context["files"].get(str(path), {})
        claimed = {name.split(".")[0] for name in entry.get("modules", [])}
        # A directory label can raise suspicion, never grant an exemption.
        for index, part in enumerate(path.parts[:-1]):
            if part in ("site-packages", "dist-packages"):
                claimed.add(path.parts[index + 1].split(".")[0])
        return (
            bool(claimed & numerical_packages)
            or entry.get("package") in numerical_packages
        )

    parents = _library_evidence(
        parent_images,
        prefix,
        lambda path: (
            path.name.lower().startswith(_COMPONENT_PREFIXES)
            or path.name.lower() in fresh_names
            or numerical_origin(path)
            or parent_context["files"].get(str(path), {}).get("valid", False)
        ),
        None if image_diagnostics is None else image_diagnostics["parent"],
        component_key=lambda path: _component_key(path, parent_context),
    )
    parent_keys = {
        _component_key(Path(item["path"]), parent_context): item for item in parents
    }
    for parent in parents:
        path = Path(parent["path"])
        key = _component_key(path, parent_context)
        if key in fresh:
            if parent == fresh[key]:
                continue
            raise RuntimeError(
                "Loaded parent and fresh-process numerical libraries disagree: "
                + parent["path"]
            )
        entry = parent_context["files"].get(str(path), {})
        package = entry.get("package")
        if (
            key[0] != "package"
            or parent["path"] in fresh_paths
            or (
                numerical_origin(path)
                and (
                    parent_context["packages"].get(package)
                    != fresh_context["packages"].get(package)
                )
            )
        ):
            raise RuntimeError(
                "Loaded parent and fresh-process numerical libraries disagree: unverified origin "
                + str(path)
            )
        runtime_key = ("runtime", path.name.lower())
        if runtime_key in fresh and (
            parent_keys.get(runtime_key) != fresh[runtime_key]
            or os.environ.get("DYLD_FORCE_FLAT_NAMESPACE") is not None
        ):
            raise RuntimeError(
                "Private helper lacks an unchanged numerical runtime: " + str(path)
            )
        if image_diagnostics is not None:
            image_diagnostics["excluded_parent"].append(
                {
                    **parent,
                    "package": package,
                    "origin": parent_context["packages"][package],
                    "modules": entry["modules"],
                    "reason": (
                        "same_numerical_installation_extra"
                        if numerical_origin(path)
                        else "separate_imported_package"
                    ),
                }
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
        "package_origins": fresh_context["packages"],
        "package_modules": {
            path: entry.get("origins", entry.get("modules", []))
            for path, entry in fresh_context["files"].items()
            if entry.get("valid") and entry.get("modules")
        },
        "libraries": libraries,
        "conda_builds": records,
        "optional_sampling_artifacts": _sampling_artifacts(),
    }
