"""Lossless records of one physical posterior and its completed trajectories.

SQLite is authoritative; status.json is only a recoverable human summary. Copy
an active calculation with ``backup`` rather than copying its database file.
Storage itself needs NumPy, but neither JAX nor the finite-element environment.
"""

# standard library imports
from contextlib import contextmanager
from dataclasses import dataclass, fields
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile

# numerical imports
import numpy as np

SCHEMA_VERSION = 1
_NUMERICAL_MODULES = (
    "heat_model",
    "forward_solver",
    "domain_coefficient",
    "temperature_observations",
    "conductivity_prior",
    "conductivity_posterior",
    "heat_inference",
    "hmc_blackjax",
    "hmc_run",
    "posterior_store",
    "native_provenance",
)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def numerical_environment():
    """Identify numerical sources, versions, precision and execution controls.

    This inspection never imports optional sampling packages. Git history is
    provenance only: continuation depends on numerical source contents.

    Returns
    -------
    dict
        Distribution and SQLite versions, source hashes, architecture, declared
        precision/backend settings, thread and MPI transport controls,
        independent of import order. Actual sampling-environment qualification
        remains separate.
    """
    import native_provenance

    versions = {}
    for name in (
        "numpy",
        "scipy",
        "fenics-dolfinx",
        "fenics-basix",
        "fenics-ffcx",
        "fenics-ufl",
        "petsc4py",
        "mpi4py",
        "jax",
        "jaxlib",
        "blackjax",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    directory = Path(__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "float64_bits": np.finfo(np.float64).bits,
        "versions": versions,
        "sqlite": sqlite3.sqlite_version,
        "native_builds": native_provenance.native_build_identity(),
        "sources": {
            name: hashlib.sha256((directory / (name + ".py")).read_bytes()).hexdigest()
            for name in _NUMERICAL_MODULES
        },
        "execution": {
            name: os.environ.get(name)
            for name in (
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "FI_PROVIDER",
                "FI_TCP_IFACE",
                "FI_SOCKETS_IFACE",
                "CC",
                "CXX",
                "CFLAGS",
                "CPPFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "LDSHARED",
                "CPATH",
                "LIBRARY_PATH",
            )
        },
    }


def source_revision():
    r"""Return checkout provenance, outside the numerical continuation identity.

    Returns
    -------
    str or None
        Git revision when available; installed source files may have no checkout.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pack(value):
    """Encode named fields as versioned JSON plus lossless NumPy .npy members."""
    arrays = {}

    def encode(item):
        if isinstance(item, (np.ndarray, np.generic)):
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise ValueError(
                    "Object arrays are not permitted in saved calculations."
                )
            key = "a" + str(len(arrays))
            arrays[key] = array
            return {"array": key, "numpy_scalar": isinstance(item, np.generic)}
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("Saved field names must be strings.")
            return {"mapping": {key: encode(val) for key, val in item.items()}}
        if isinstance(item, (list, tuple)):
            return {
                "sequence": [encode(val) for val in item],
                "tuple": isinstance(item, tuple),
            }
        if item is None or type(item) in (str, int, float, bool):
            # Nonfinite diagnostic statistics are represented explicitly, rather
            # than relying on nonstandard NaN/Infinity JSON tokens.
            if isinstance(item, float) and not np.isfinite(item):
                return {"nonfinite_float": repr(item)}
            return {"scalar": item}
        raise TypeError(f"Unsupported saved field type: {type(item).__name__}.")

    description = {"schema_version": SCHEMA_VERSION, "fields": encode(value)}
    arrays["description"] = np.frombuffer(_json(description), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        description = json.loads(archive["description"].tobytes())
        if description["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported saved state schema version.")

        def decode(item):
            if "array" in item:
                value = archive[item["array"]].copy()
                return value[()] if item["numpy_scalar"] else value
            if "mapping" in item:
                return {key: decode(val) for key, val in item["mapping"].items()}
            if "sequence" in item:
                result = [decode(val) for val in item["sequence"]]
                return tuple(result) if item["tuple"] else result
            if "nonfinite_float" in item:
                return float(item["nonfinite_float"])
            return item["scalar"]

        return decode(description["fields"])


def _copy(value):
    return _unpack(_pack(value))


@dataclass(frozen=True)
class ProblemRecord:
    """Frozen finite-dimensional physical law and complete numerical identity.

    ``metadata`` describes evaluated physical inputs and their units; ``arrays``
    retain the complete nodal mesh/fields. No original callable is needed.

    Parameters
    ----------
    metadata : dict
        JSON-compatible physical declarations, units, prior and solver controls.
    arrays : dict of str to numpy.ndarray
        Lossless finite real model arrays, with no object arrays.
    problem_id : str
        Identity of the physical model, observations and probability law.
    numerical_id : str
        Identity of fixed numerical settings, sources and environment.
    """

    metadata: dict
    arrays: dict[str, np.ndarray]
    problem_id: str
    numerical_id: str

    def __post_init__(self):
        metadata = json.loads(_json(self.metadata))
        arrays = {}
        for name, value in self.arrays.items():
            if not isinstance(name, str):
                raise TypeError("Problem array names must be strings.")
            array = np.asarray(value)
            if array.dtype.hasobject or array.dtype.kind not in "biuf":
                raise ValueError("Physical model arrays must be real numerical arrays.")
            if not np.isfinite(array).all():
                raise ValueError("Physical model arrays must be finite.")
            arrays[name] = array.copy()
        for name in ("problem_id", "numerical_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty.")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "arrays", arrays)


@dataclass(frozen=True)
class ChainCheckpoint:
    """State at a completed trajectory boundary, including the next random key.

    ``phase`` labels the preceding completed transition. Before any transition,
    it labels the initial phase. Completed counts select the next phase, so the
    warmup/sampling boundary needs no separate checkpoint mutation. ``step_size``
    is the next spacing; the preceding spacing belongs in trajectory statistics.

    Parameters
    ----------
    chain : int
        Declared chain identifier.
    phase : {"warmup", "sample"}
        Preceding completed phase, or the initial phase before any transitions.
    warmup_done, draws_done : int
        Completed counts, with retained draws counted separately from warmup.
    inferred_coordinates : dict
        Every declared inferred coordinate, including future auxiliary fields.
    physical_state : dict
        Cached positive ``h`` and ``sensor_temperature`` arrays, avoiding a
        repeated evaluation of the current state on continuation.
    hmc_state, rng_state, adaptation_state : dict
        Complete explicit named arrays/scalars from the sampling connection;
        include random implementation/key data and every adaptation field.
    step_size : float
        Positive physical spacing available for the next iteration.
    counters : dict
        Nonnegative cumulative integer diagnostics at this same boundary.
    """

    chain: int
    phase: str
    warmup_done: int
    draws_done: int
    inferred_coordinates: dict
    physical_state: dict
    hmc_state: dict
    rng_state: dict
    adaptation_state: dict
    step_size: float
    counters: dict


@dataclass(frozen=True)
class TrajectoryRecord:
    r"""One completed transition, retaining the previous field upon rejection.

    Parameters
    ----------
    chain : int
        Declared chain identifier.
    phase : {"warmup", "sample"}
        Completed transition's phase.
    index : int
        Zero-based index within this chain and phase.
    inferred_coordinates : dict
        Complete retained inferred state after selection.
    h : numpy.ndarray
        Positive conductivity at every canonical physical node.
    sensor_temperature : numpy.ndarray
        Predicted temperatures, in K, at the fixed observation locations.
    statistics : dict
        Explicit transition diagnostics, including the spacing actually used
        when recorded; this can differ from the next adapted checkpoint spacing.
    """

    chain: int
    phase: str
    index: int
    inferred_coordinates: dict
    h: np.ndarray
    sensor_temperature: np.ndarray
    statistics: dict


def _fields(instance):
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


def _integer(value, name, *, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numerical values.")
    return array


def _same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    return left == right


def _same_structure(initial, current):
    """Retain complete named continuation fields without library-specific names.

    Numerical scalar leaves may change between Python scalars and NumPy shape
    () arrays during adaptation. Array shapes and nonscalar dtypes remain fixed.
    """
    if isinstance(initial, dict):
        return (
            isinstance(current, dict)
            and initial.keys() == current.keys()
            and all(_same_structure(initial[key], current[key]) for key in initial)
        )
    if isinstance(initial, (list, tuple)):
        return (
            isinstance(current, (list, tuple))
            and len(initial) == len(current)
            and all(_same_structure(a, b) for a, b in zip(initial, current))
        )
    if isinstance(initial, str):
        return isinstance(current, str) and initial == current
    if initial is None:
        return current is None
    numeric = (np.ndarray, np.generic, int, float, bool)
    if not isinstance(initial, numeric) or not isinstance(current, numeric):
        return False
    initial, current = np.asarray(initial), np.asarray(current)
    return initial.shape == current.shape and (
        initial.shape == () or initial.dtype == current.dtype
    )


def _atomic_file(path, payload):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _problem_bytes(problem):
    stream = io.BytesIO()
    np.savez(stream, **problem.arrays)
    arrays = stream.getvalue()
    document = {
        "schema_version": SCHEMA_VERSION,
        "metadata": problem.metadata,
        "problem_id": problem.problem_id,
        "numerical_id": problem.numerical_id,
        "arrays_sha256": hashlib.sha256(arrays).hexdigest(),
    }
    document["record_sha256"] = hashlib.sha256(_json(document)).hexdigest()
    return _json(document), arrays


def _problem_content_digest(problem):
    """Bind a database to its exact immutable physical/numerical record."""
    digest = hashlib.sha256(
        _json(
            {
                "metadata": problem.metadata,
                "problem_id": problem.problem_id,
                "numerical_id": problem.numerical_id,
            }
        )
    )
    for name in sorted(problem.arrays):
        array = problem.arrays[name]
        digest.update(
            _json({"name": name, "dtype": array.dtype.str, "shape": array.shape})
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_problem(path):
    document = json.loads((path / "problem.json").read_bytes())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported problem schema version.")
    claimed = document.pop("record_sha256", None)
    if claimed != hashlib.sha256(_json(document)).hexdigest():
        raise ValueError("Problem record integrity hash mismatch.")
    payload = (path / "problem.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != document["arrays_sha256"]:
        raise ValueError("Physical array integrity hash mismatch.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return ProblemRecord(
        document["metadata"], arrays, document["problem_id"], document["numerical_id"]
    )


class RunStore:
    """One calculation, with a single writer and portable read-only inspection."""

    @classmethod
    def create(cls, path, *, problem, run_metadata):
        r"""Create and own a new saved physical calculation.

        Parameters
        ----------
        path : path-like
            New calculation directory, created exclusively. Existing paths are
            refused; interrupted creation evidence is preserved.
        problem : ProblemRecord
            Complete frozen physical law, arrays and numerical identity. A
            declared ``numerical_environment`` must match the current observation.
        run_metadata : dict
            Required positive integer ``chains`` and per-chain ``total_draws``;
            nonnegative integer ``warmup`` defaults to zero. Seed, controls,
            initialization recipe and other named provenance are retained.
            The checkout revision is recorded separately from source identity.

        Returns
        -------
        RunStore
            Writable calculation owning the single-writer lock. Close it or use
            a context manager to release the lock and database connection."""
        path = Path(path).resolve()
        # Validate before claiming a directory; a failed creation thereafter is
        # deliberately distinguishable from a completed empty calculation.
        problem = ProblemRecord(
            problem.metadata, problem.arrays, problem.problem_id, problem.numerical_id
        )
        metadata = _copy(run_metadata)
        _integer(metadata["total_draws"], "total_draws", minimum=1)
        _integer(metadata["chains"], "chains", minimum=1)
        _integer(metadata.get("warmup", 0), "warmup")
        path.mkdir()
        instance = cls.__new__(cls)
        instance._connection = instance._lock = None
        instance._path, instance._writable = path, True
        try:
            instance._take_lock()
            document, arrays = _problem_bytes(problem)
            _atomic_file(path / "problem.npz", arrays)
            _atomic_file(path / "problem.json", document)
            instance._connection = sqlite3.connect(
                path / "chains.sqlite", isolation_level=None
            )
            instance._configure()
            instance._connection.executescript("""
                CREATE TABLE run (id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL, metadata BLOB NOT NULL, environment BLOB NOT NULL, initialized INTEGER NOT NULL, complete INTEGER NOT NULL, problem_id TEXT NOT NULL, numerical_id TEXT NOT NULL, problem_sha256 TEXT NOT NULL);
                CREATE TABLE checkpoints (chain INTEGER PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE initial_checkpoints (chain INTEGER PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE trajectories (chain INTEGER NOT NULL, phase TEXT NOT NULL, iteration INTEGER NOT NULL, data BLOB NOT NULL, PRIMARY KEY(chain,phase,iteration));
                CREATE TABLE counters (chain INTEGER PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE failures (id INTEGER PRIMARY KEY, data BLOB NOT NULL);
            """)
            metadata.setdefault("git_revision", source_revision())
            environment = numerical_environment()
            with instance._transaction():
                instance._connection.execute(
                    "INSERT INTO run VALUES(1,?,?,?,?,0,?,?,?)",
                    (
                        SCHEMA_VERSION,
                        _pack(metadata),
                        _pack(environment),
                        0,
                        problem.problem_id,
                        problem.numerical_id,
                        _problem_content_digest(problem),
                    ),
                )
            # Preserve both observations if creation fails; only consistent
            # numerical provenance may become a completed calculation.
            if "numerical_environment" in problem.metadata and not _same(
                problem.metadata["numerical_environment"], environment
            ):
                raise ValueError(
                    "Declared numerical environment differs from the current "
                    "observation; run creation is incomplete."
                )
            with instance._transaction():
                instance._connection.execute("UPDATE run SET complete=1 WHERE id=1")
            instance._problem = problem
            instance._write_status()
            return instance
        except BaseException:
            instance.close()
            raise

    @classmethod
    def open(cls, path, *, writable=False):
        r"""Open a completed calculation for inspection or continuation.

        Parameters
        ----------
        path : path-like
            Existing calculation with its original problem files and database.
        writable : bool, optional
            Defaults to False. True acquires the exclusive writer lock, verifies
            current numerical sources/environment, and rebuilds advisory status.
            Both modes verify the database's frozen physical-record binding.

        Returns
        -------
        RunStore
            Calculation handle. Read-only inspection needs NumPy and does not
            require the saved sampling dependencies or their environment."""
        instance = cls.__new__(cls)
        instance._connection = instance._lock = None
        instance._path, instance._writable = Path(path).resolve(), bool(writable)
        try:
            instance._problem = _read_problem(instance._path)
            if writable:
                instance._take_lock()
            uri = (instance._path / "chains.sqlite").as_uri() + (
                "?mode=rw" if writable else "?mode=ro"
            )
            instance._connection = sqlite3.connect(uri, uri=True, isolation_level=None)
            row = instance._connection.execute(
                "SELECT schema_version,environment,complete,problem_id,numerical_id,problem_sha256 FROM run WHERE id=1"
            ).fetchone()
            if row is None or row[0] != SCHEMA_VERSION or row[2] != 1:
                raise ValueError("Unsupported schema or incomplete run creation.")
            if (
                row[3] != instance._problem.problem_id
                or row[4] != instance._problem.numerical_id
                or row[5] != _problem_content_digest(instance._problem)
            ):
                raise ValueError("Database/problem identity and integrity mismatch.")
            if writable:
                if not _same(_unpack(row[1]), numerical_environment()):
                    raise ValueError(
                        "Changed numerical sources or environment cannot append to this calculation."
                    )
                instance._configure()
                instance._write_status()
            else:
                instance._connection.execute("PRAGMA query_only=ON")
            return instance
        except BaseException:
            instance.close()
            raise

    @property
    def path(self):
        """Resolved directory holding this physical calculation."""
        return self._path

    def failures(self):
        """Read saved interruption and fatal-solve evidence in event order.

        Returns
        -------
        iterator of dict
            Copied failure evidence, including separately abandoned solve cost.
        """
        for row in self._connection.execute("SELECT data FROM failures ORDER BY id"):
            yield _unpack(row[0])

    def _take_lock(self):
        self._lock = (self._path / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError(
                "This calculation already has an active writer."
            ) from None

    def _configure(self):
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")

    def _require_writer(self):
        if self._connection is None:
            raise RuntimeError("The calculation is closed.")
        if not self._writable:
            raise PermissionError("This calculation was opened read-only.")

    @contextmanager
    def _transaction(self):
        self._require_writer()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    @property
    def metadata(self):
        r"""Return copied calculation controls and provenance.

        Returns
        -------
        dict
            Current per-chain draw target, chain count, warmup count, and saved
            caller metadata. Mutating this copy does not change the calculation."""
        return _unpack(
            self._connection.execute("SELECT metadata FROM run WHERE id=1").fetchone()[
                0
            ]
        )

    @property
    def problem_record(self):
        r"""Return a copied frozen physical experiment.

        Returns
        -------
        ProblemRecord
            Complete physical arrays, metadata and identities, independent of
            the store's immutable experiment record."""
        p = self._problem
        return ProblemRecord(p.metadata, p.arrays, p.problem_id, p.numerical_id)

    def validate_identity(self, problem):
        r"""Verify a proposed calculation describes this saved posterior.

        Parameters
        ----------
        problem : ProblemRecord
            Proposed frozen physical/numerical identities and exact inputs.

        Returns
        -------
        None
            Raises ValueError on changed identities or contents. Writable open
            separately compares the current numerical sources/environment."""
        if (
            problem.problem_id != self._problem.problem_id
            or problem.numerical_id != self._problem.numerical_id
        ):
            raise ValueError(
                "Physical problem or numerical identity differs from the saved calculation."
            )
        if not _same(problem.metadata, self._problem.metadata) or not _same(
            problem.arrays, self._problem.arrays
        ):
            raise ValueError(
                "Physical problem contents differ from the saved calculation."
            )

    def _validate_checkpoint(self, cp):
        _integer(cp.chain, "chain")
        if cp.chain >= self.metadata["chains"]:
            raise ValueError("Checkpoint chain lies outside the declared calculation.")
        if cp.phase not in ("warmup", "sample"):
            raise ValueError("Unknown trajectory phase.")
        _integer(cp.warmup_done, "warmup_done")
        _integer(cp.draws_done, "draws_done")
        if cp.draws_done > self.metadata[
            "total_draws"
        ] or cp.warmup_done > self.metadata.get("warmup", 0):
            raise ValueError("Checkpoint counts exceed the requested calculation.")
        schema = self._problem.metadata.get("coordinate_schema")
        if not isinstance(cp.inferred_coordinates, dict) or not cp.inferred_coordinates:
            raise ValueError("Complete inferred coordinates are required.")
        if schema is not None and set(schema) != set(cp.inferred_coordinates):
            raise ValueError("Checkpoint must retain every inferred coordinate.")
        for name, value in cp.inferred_coordinates.items():
            array = _finite_array(value, "inferred coordinates")
            if schema is not None and array.shape != tuple(schema[name]):
                raise ValueError(
                    "Inferred coordinate length disagrees with the physical problem."
                )
        h = _finite_array(cp.physical_state["h"], "conductivity")
        temperature = _finite_array(
            cp.physical_state["sensor_temperature"], "sensor temperature"
        )
        if h.ndim != 1 or np.any(h <= 0) or temperature.ndim != 1:
            raise ValueError(
                "Physical checkpoint fields have invalid shape or conductivity."
            )
        if "coordinates" in self._problem.arrays and len(h) != len(
            self._problem.arrays["coordinates"]
        ):
            raise ValueError("Conductivity length disagrees with the physical mesh.")
        if "sensor_values" in self._problem.arrays and len(temperature) != len(
            self._problem.arrays["sensor_values"]
        ):
            raise ValueError(
                "Temperature length disagrees with the observation vector."
            )
        if (
            np.ndim(cp.step_size) != 0
            or not np.isfinite(cp.step_size)
            or cp.step_size <= 0
        ):
            raise ValueError("A positive finite trajectory step size is required.")
        for name in ("hmc_state", "rng_state", "adaptation_state", "counters"):
            if not isinstance(getattr(cp, name), dict):
                raise ValueError(f"{name} must contain explicit named fields.")

        def finite_state(value):
            if isinstance(value, dict):
                for item in value.values():
                    finite_state(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    finite_state(item)
            elif isinstance(
                value, (np.ndarray, np.generic, int, float)
            ) and not isinstance(value, bool):
                _finite_array(value, "continuation state")

        for state in (cp.hmc_state, cp.rng_state, cp.adaptation_state):
            finite_state(state)
        if "position" in cp.hmc_state and "z" in cp.inferred_coordinates:
            if not _same(cp.hmc_state["position"], cp.inferred_coordinates["z"]):
                raise ValueError(
                    "HMC position and retained inferred coordinates disagree."
                )
        for value in cp.counters.values():
            _integer(value, "diagnostic counter")
        _pack(_fields(cp))

    def initialize_chains(self, checkpoints):
        r"""Commit every actual initial state together, exactly once.

        Parameters
        ----------
        checkpoints : sequence of ChainCheckpoint
            Exactly one complete state per declared chain, with zero completed
            warmup and retained counts. Initial phase is warmup when requested,
            otherwise sample. Include every HMC, random and adaptation field;
            subsequent checkpoints preserve this named structure and shapes.

        Returns
        -------
        None
            Initial evidence, mutable continuation states and counters commit
            atomically. ``initial_checkpoint`` retains each original state."""
        self._require_writer()
        checkpoints = list(checkpoints)
        if sorted(cp.chain for cp in checkpoints) != list(
            range(self.metadata["chains"])
        ):
            raise ValueError("Initialize exactly one state for every declared chain.")
        for cp in checkpoints:
            self._validate_checkpoint(cp)
            expected_phase = "warmup" if self.metadata.get("warmup", 0) else "sample"
            if cp.phase != expected_phase:
                raise ValueError("Initial phase disagrees with the requested warmup.")
            if cp.draws_done != 0 or cp.warmup_done != 0:
                raise ValueError(
                    "Initial checkpoints must precede all warmup and retained trajectories."
                )
        with self._transaction():
            if self._connection.execute(
                "SELECT initialized FROM run WHERE id=1"
            ).fetchone()[0]:
                raise RuntimeError("Chain initialization has already been committed.")
            for cp in checkpoints:
                payload = _pack(_fields(cp))
                self._connection.execute(
                    "INSERT INTO initial_checkpoints VALUES(?,?)",
                    (int(cp.chain), payload),
                )
                self._connection.execute(
                    "INSERT INTO checkpoints VALUES(?,?)",
                    (int(cp.chain), payload),
                )
                self._connection.execute(
                    "INSERT INTO counters VALUES(?,?)",
                    (int(cp.chain), _pack(cp.counters)),
                )
            self._connection.execute("UPDATE run SET initialized=1 WHERE id=1")
        self._write_status()

    def checkpoint(self, chain):
        r"""Restore the last completed physical trajectory boundary.

        Parameters
        ----------
        chain : int
            Declared chain identifier.

        Returns
        -------
        ChainCheckpoint
            Copied inferred/physical fields, current HMC state, next random key,
            next spacing, adaptation and cumulative counters. Completed counts
            determine what remains. Missing initial states raise KeyError."""
        row = self._connection.execute(
            "SELECT data FROM checkpoints WHERE chain=?", (_integer(chain, "chain"),)
        ).fetchone()
        if row is None:
            raise KeyError(chain)
        return ChainCheckpoint(**_unpack(row[0]))

    def initial_checkpoint(self, chain):
        r"""Recover the complete original physical, inferred and random state.

        Parameters
        ----------
        chain : int
            Declared chain identifier.

        Returns
        -------
        ChainCheckpoint
            Immutable saved start, copied for the caller. Advancing the mutable
            continuation checkpoint never replaces this initial evidence.
        """
        row = self._connection.execute(
            "SELECT data FROM initial_checkpoints WHERE chain=?",
            (_integer(chain, "chain"),),
        ).fetchone()
        if row is None:
            raise KeyError(chain)
        return ChainCheckpoint(**_unpack(row[0]))

    def commit_trajectory(self, record, checkpoint):
        r"""Commit one physical outcome and its complete continuation state.

        Parameters
        ----------
        record : TrajectoryRecord
            Completed warmup or sample transition at the next zero-based index.
            Rejections retain the preceding field. Statistics describe this
            transition, including its spacing when recorded.
        checkpoint : ChainCheckpoint
            Matching retained coordinates/physical predictions, updated counters
            and adaptation, and the next random key/spacing. Phase labels the
            completed transition; counts determine the next phase. The next
            warmup spacing may differ from the spacing in record statistics.

        Returns
        -------
        None
            Record, checkpoint and counters are committed in one transaction.
            Inconsistent indices, lengths, fields or counts are rejected."""
        self._require_writer()
        cp = checkpoint
        self._validate_checkpoint(cp)
        _integer(record.chain, "trajectory chain")
        _integer(record.index, "trajectory index")
        if record.chain != cp.chain or record.phase != cp.phase:
            raise ValueError("Trajectory and checkpoint chain/phase disagree.")
        old = self.checkpoint(cp.chain)
        initial = self.initial_checkpoint(cp.chain)
        for name in ("hmc_state", "rng_state", "adaptation_state"):
            if not _same_structure(getattr(initial, name), getattr(cp, name)):
                raise ValueError(
                    "Complete continuation mapping fields or shapes changed."
                )
        warmup = record.phase == "warmup"
        previous = old.warmup_done if warmup else old.draws_done
        if (
            record.index != previous
            or (cp.warmup_done if warmup else cp.draws_done) != previous + 1
        ):
            raise ValueError("Trajectory index and checkpoint counts disagree.")
        if (
            cp.draws_done != old.draws_done
            if warmup
            else cp.warmup_done != old.warmup_done
        ):
            raise ValueError("A trajectory may advance only its own phase count.")
        if warmup and old.phase == "sample":
            raise ValueError("Retained sampling cannot return to warmup.")
        if not warmup and cp.warmup_done != self.metadata.get("warmup", 0):
            raise ValueError("Retained draws require completed warmup.")
        for current, previous_value in (
            (cp.inferred_coordinates, old.inferred_coordinates),
            (cp.physical_state, old.physical_state),
        ):
            if current.keys() != previous_value.keys() or any(
                np.shape(current[key]) != np.shape(previous_value[key])
                for key in current
            ):
                raise ValueError("Continuation field names or lengths changed.")
        if (
            not _same(record.inferred_coordinates, cp.inferred_coordinates)
            or not _same(record.h, cp.physical_state["h"])
            or not _same(
                record.sensor_temperature, cp.physical_state["sensor_temperature"]
            )
        ):
            raise ValueError(
                "Trajectory physical fields and checkpoint do not correspond."
            )
        if cp.counters.keys() != old.counters.keys() or any(
            cp.counters[key] < old.counters[key] for key in cp.counters
        ):
            raise ValueError(
                "Diagnostic counters must preserve every cumulative count."
            )
        record_payload, checkpoint_payload = _pack(_fields(record)), _pack(_fields(cp))
        with self._transaction():
            # The writer lock excludes other processes; this guards stale callers
            # attempting to reuse an earlier boundary within this writer.
            count = self._connection.execute(
                "SELECT COUNT(*) FROM trajectories WHERE chain=? AND phase=?",
                (int(cp.chain), cp.phase),
            ).fetchone()[0]
            if count != record.index:
                raise ValueError(
                    "Stored trajectory length and proposed index disagree."
                )
            self._connection.execute(
                "INSERT INTO trajectories VALUES(?,?,?,?)",
                (int(record.chain), record.phase, int(record.index), record_payload),
            )
            self._connection.execute(
                "UPDATE checkpoints SET data=? WHERE chain=?",
                (checkpoint_payload, int(cp.chain)),
            )
            self._connection.execute(
                "UPDATE counters SET data=? WHERE chain=?",
                (_pack(cp.counters), int(cp.chain)),
            )
        self._write_status()

    def records(self, chain, *, phase="sample", start=0, stop=None):
        r"""Read an ordered slice of physical trajectory outcomes.

        Parameters
        ----------
        chain : int
            Declared chain identifier.
        phase : {"sample", "warmup"}, optional
            Defaults to retained samples; warmup remains a separate sequence.
        start : int, optional
            Nonnegative first zero-based index, inclusive; defaults to zero.
        stop : int or None, optional
            Exclusive upper index. None reads through the committed end.

        Returns
        -------
        iterator of TrajectoryRecord
            Lazy physical outcomes in index order, including every rejection
            repeat. Calling this method does not load all draws at once."""
        chain = _integer(chain, "chain")
        start = _integer(start, "start")
        if phase not in ("warmup", "sample"):
            raise ValueError("Unknown trajectory phase.")
        if stop is None:
            stop = 2**63 - 1
        stop = _integer(stop, "stop")
        if stop < start:
            raise ValueError("stop precedes start.")
        cursor = self._connection.execute(
            "SELECT data FROM trajectories WHERE chain=? AND phase=? AND iteration>=? AND iteration<? ORDER BY iteration",
            (chain, phase, start, stop),
        )
        for row in cursor:
            yield TrajectoryRecord(**_unpack(row[0]))

    def update_target(self, total_draws):
        r"""Extend the requested retained total while preserving the calculation.

        Parameters
        ----------
        total_draws : int
            New positive per-chain retained target, at least the current target.
            Equal targets are harmless; reductions are rejected.

        Returns
        -------
        None
            Only the requested total changes. Physical law, controls, chain
            identities, initial states and continuation states are preserved."""
        self._require_writer()
        total_draws = _integer(total_draws, "total_draws", minimum=1)
        with self._transaction():
            metadata = self.metadata
            if total_draws < metadata["total_draws"]:
                raise ValueError("An extension cannot reduce the requested total.")
            metadata["total_draws"] = total_draws
            self._connection.execute(
                "UPDATE run SET metadata=? WHERE id=1", (_pack(metadata),)
            )
        self._write_status()

    def write_failure(self, evidence):
        r"""Preserve fatal physical-solve evidence beside the last complete state.

        Parameters
        ----------
        evidence : dict
            Explicit named scalars/arrays describing the failure, for example
            chain, phase, proposed coordinates, solve reason and residual.

        Returns
        -------
        None
            Evidence commits independently; no trajectory or continuation count
            advances, and the previous complete checkpoint remains available."""
        with self._transaction():
            self._connection.execute(
                "INSERT INTO failures(data) VALUES(?)", (_pack(evidence),)
            )
        self._write_status()

    def _write_status(self):
        summary = {
            "schema_version": SCHEMA_VERSION,
            "total_draws": int(self.metadata["total_draws"]),
            "chains": {},
        }
        for chain, data in self._connection.execute(
            "SELECT chain,data FROM checkpoints ORDER BY chain"
        ):
            cp = _unpack(data)
            summary["chains"][str(chain)] = {
                "phase": cp["phase"],
                "warmup_done": int(cp["warmup_done"]),
                "draws_done": int(cp["draws_done"]),
                "counters": {
                    name: int(value) for name, value in cp["counters"].items()
                },
            }
        failures = list(self.failures())
        if failures:
            last = failures[-1]
            summary["last_failure"] = {
                name: last.get(name)
                for name in (
                    "output",
                    "chain",
                    "phase",
                    "trajectory",
                    "exception",
                    "message",
                    "committed",
                )
            }
        summary["aborted_work"] = {
            name: sum(item.get("aborted_work", {}).get(name, 0) for item in failures)
            for name in ("heat", "adjoint", "heat_iterations", "adjoint_iterations")
        }
        try:
            _atomic_file(self._path / "status.json", _json(summary))
        except OSError:
            # A stale summary must never turn a durable trajectory into a
            # reported failed transition that a caller might then repeat.
            pass

    def backup(self, destination):
        r"""Save a coherent portable snapshot, including committed WAL contents.

        Parameters
        ----------
        destination : path-like
            New snapshot directory. Existing destinations are refused.

        Returns
        -------
        None
            Frozen problem files, SQLite Connection.backup snapshot, and rebuilt
            advisory status are saved. Copying an active database file alone
            does not provide this consistency guarantee."""
        destination = Path(destination).resolve()
        destination.mkdir()
        target = None
        try:
            document, arrays = _problem_bytes(self._problem)
            _atomic_file(destination / "problem.json", document)
            _atomic_file(destination / "problem.npz", arrays)
            target = sqlite3.connect(destination / "chains.sqlite")
            self._connection.backup(target)
            target.close()
            target = None
            # Rebuild summary from the snapshot rather than racing current draws.
            with RunStore.open(destination) as snapshot:
                snapshot._write_status()
        finally:
            if target is not None:
                target.close()

    def close(self):
        r"""Release this calculation's database and exclusive writer lock.

        Returns
        -------
        None
            Repeated calls are harmless. The physical experiment and all durable
            trajectory/checkpoint evidence remain available for reopening."""
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None
        lock = getattr(self, "_lock", None)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            self._lock = None

    def __enter__(self):
        if self._connection is None:
            raise RuntimeError("The calculation is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        self.close()
