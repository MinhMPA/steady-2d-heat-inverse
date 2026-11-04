"""Fresh-process, indexed calibration ownership; numerical imports follow locking.

This module keeps execution attempts separate from the physical experiment.
An absent lock never makes an unclosed attempt safe to repeat.
"""

from contextlib import contextmanager
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

_RESOURCE_ERRNOS = frozenset(
    {
        errno.ENOSPC,
        errno.EDQUOT,
        errno.ENOMEM,
        errno.EMFILE,
        errno.ENFILE,
    }
)


class ReplicaBusy(RuntimeError):
    """A replica already has an owner."""


def atomic_json(path, value):
    """Publish a complete local record and flush its containing directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(path):
    return json.loads(Path(path).read_text())


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _binding(output, replicates, identity):
    manifest = _read(Path(output) / "manifest.json")
    actual = hashlib.sha256(
        json.dumps(manifest["identity"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest["identity_sha256"] != actual
        or identity != actual
        or manifest["identity"]["kind"] != "sbc"
        or manifest["identity"]["extra"]["replicates"] != replicates
    ):
        raise ValueError("Study manifest, fixed total or identity changed.")
    return manifest


@contextmanager
def _replica_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "replica.lock").open("a") as stream:
        os.set_inheritable(stream.fileno(), False)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReplicaBusy(str(directory)) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def issue_attempt(output, *, index, replicates, identity, session, command):
    """Durably reserve an index before the parent can create its process."""
    output = Path(output).resolve()
    _integer(replicates, "replicates", 1)
    _integer(index, "index")
    if index >= replicates:
        raise ValueError("Replica index is outside the fixed study.")
    _binding(output, replicates, identity)
    if not session or Path(session).name != session:
        raise ValueError("Execution session must be one path component.")
    attempt = uuid.uuid4().hex
    path = (
        output / "executions" / session / f"attempt-{index:03d}-{attempt}.issued.json"
    )
    atomic_json(
        path,
        {
            "schema": 1,
            "attempt": attempt,
            "replicate": index,
            "replicates": replicates,
            "identity": identity,
            "issued_unix": time.time(),
            "command": list(command),
        },
    )
    return path


def _issues(output, index, replicates, identity):
    result = []
    for path in sorted((output / "executions").glob("*/attempt-*.issued.json")):
        issue = _read(path)
        if (
            issue.get("replicates") != replicates
            or issue.get("identity") != identity
            or type(issue.get("replicate")) is not int
            or not 0 <= issue["replicate"] < replicates
            or not isinstance(issue.get("attempt"), str)
            or len(issue["attempt"]) != 32
            or any(c not in "0123456789abcdef" for c in issue["attempt"])
        ):
            raise ValueError("Invalid or changed durable attempt record.")
        if issue["replicate"] == index:
            result.append((path, issue))
    result.sort(key=lambda item: (item[1]["issued_unix"], str(item[0])))
    return result


def _closure_path(directory, issue):
    return directory / f"attempt-{issue['attempt']}.closure.json"


def _sidecar(path, kind):
    return path.with_name(path.name.replace(".issued.json", f".{kind}.json"))


def _same_attempt(record, issue):
    return all(
        record.get(k) == issue[k]
        for k in ("attempt", "replicate", "replicates", "identity")
    )


def _closure(directory, path, issue):
    closure = _closure_path(directory, issue)
    if closure.exists():
        recorded = _read(closure)
        if not _same_attempt(recorded, issue):
            raise ValueError("Attempt closure does not match its durable issue.")
        return recorded
    observed = _sidecar(path, "exit")
    if observed.exists():
        recorded = _read(observed)
        if not _same_attempt(recorded, issue):
            raise ValueError("Process exit does not match its durable issue.")
        # Exit zero alone is not terminal scientific or cooperative evidence.
        return {**recorded, "status": "unresolved", "cooperative": False}
    return None


def inspect_replica(
    output, *, index, replicates, identity, validate_result, validate_resume=None
):
    """Reserve pending starts; validate terminal/resumable evidence under lock."""
    output = Path(output).resolve()
    _binding(output, replicates, identity)
    _integer(index, "index")
    if index >= replicates:
        raise ValueError("Replica index is outside the fixed study.")
    directory = output / f"replicate-{index:03d}"
    state = {
        "replicate": index,
        "eligible": False,
        "reserved": False,
        "status": "unresolved",
    }
    try:
        issues = _issues(output, index, replicates, identity)
        if issues:
            latest_path, latest = issues[-1]
            state["attempt"] = str(latest_path)
            started = directory / f"attempt-{latest['attempt']}.started.json"
            if (
                not started.exists()
                and not (directory / "result.json").exists()
                and _closure(directory, latest_path, latest) is None
            ):
                # The legitimate worker still needs its first ownership lock.
                # Inspecting must not briefly take it and force that worker to
                # exit as a duplicate. Missing evidence only reserves capacity.
                state.update(reserved=True, reason="issued_attempt_unclosed")
                return state
        with _replica_lock(directory):
            terminal = directory / "result.json"
            if terminal.exists():
                result = _read(terminal)
                matches = [(p, i) for p, i in issues if _same_attempt(result, i)]
                if len(matches) != 1 or result.get("status") not in (
                    "qualified",
                    "failed",
                ):
                    raise ValueError(
                        "Terminal result has no unique matching issued attempt."
                    )
                matched_path, matched = matches[0]
                for path, issue in issues:
                    if issue["attempt"] == matched["attempt"]:
                        continue
                    closed = _closure(directory, path, issue)
                    if not (
                        closed
                        and closed.get("status") == "interrupted"
                        and closed.get("cooperative") is True
                    ):
                        raise ValueError(
                            "Other attempt remains unclosed or unresolved."
                        )
                validate_result(directory, result)
                # Terminal publication is authoritative even if death prevented
                # the worker's separate closure. No new scientific attempt follows.
                state.update(status=result["status"], result=result)
                return state
            if not issues:
                unexpected = [
                    p.name for p in directory.iterdir() if p.name != "replica.lock"
                ]
                if unexpected:
                    raise ValueError(
                        "Replica artifacts exist without an issued attempt."
                    )
                state.update(status="not_started", eligible=True)
                return state
            for path, issue in issues:
                closed = _closure(directory, path, issue)
                if closed is None:
                    state.update(
                        reserved=True,
                        attempt=str(path),
                        reason="issued_attempt_unclosed",
                    )
                    return state
                if not (
                    closed.get("status") == "interrupted"
                    and closed.get("cooperative") is True
                ):
                    state.update(
                        reason="attempt_unresolved_or_terminal_record_missing",
                        closure=closed,
                    )
                    return state
            if validate_resume is not None:
                validate_resume(directory)
            state.update(status="interrupted", eligible=True)
            return state
    except ReplicaBusy:
        state.update(status="active", reserved=True, reason="replica_lock_held")
        return state
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(reason=str(exc), reserved=state.get("reserved", False))
        # A corrupt/unreadable issue cannot establish that a launched worker ended.
        state["reserved"] = True
        return state


def _launch(command, log):
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        env=environment,
    )


def coordinate(
    output,
    *,
    replicates,
    workers,
    identity,
    validate_result,
    validate_resume=None,
    command=None,
    event=None,
    progress=None,
    poll_interval=0.2,
):
    """Parent-only bounded scheduling; caller holds the study lock throughout."""
    output = Path(output).resolve()
    _integer(replicates, "replicates", 1)
    _integer(workers, "workers", 1)
    if workers > replicates:
        raise ValueError("workers cannot exceed the fixed replica total.")
    _binding(output, replicates, identity)
    command = (
        [sys.executable, "-m", "hmc_sbc_worker"] if command is None else list(command)
    )
    session = uuid.uuid4().hex
    session_path = output / "executions" / session
    atomic_json(
        session_path / "session.json",
        {
            "workers": workers,
            "replicates": replicates,
            "identity": identity,
            "interpreter": sys.executable,
            "started_unix": time.time(),
            "parent_pid": os.getpid(),
        },
    )
    owned, attempted = {}, set()
    validated_results = {}

    def validate_completed(directory, result):
        # Complete chain databases are immutable here. Avoid rereading every
        # completed database at every scheduling poll; revalidate at collection.
        key = str(directory)
        if validated_results.get(key) != result:
            validate_result(directory, result)
            validated_results[key] = result
        return result

    stopping = False
    interrupted = False

    def emit(kind, **fields):
        if event is not None:
            event(kind, **fields)

    def exits():
        for index, (process, issued, issue, log) in list(owned.items()):
            code = process.poll()
            if code is not None:
                log.close()
                atomic_json(
                    _sidecar(issued, "exit"),
                    {
                        **{
                            k: issue[k]
                            for k in ("attempt", "replicate", "replicates", "identity")
                        },
                        "returncode": code,
                        "pid": process.pid,
                        "exited_unix": time.time(),
                    },
                )
                emit(
                    "replica_process_exit",
                    replicate=index,
                    attempt=issue["attempt"],
                    returncode=code,
                )
                del owned[index]

    try:
        while True:
            try:
                exits()
                states = [
                    inspect_replica(
                        output,
                        index=i,
                        replicates=replicates,
                        identity=identity,
                        validate_result=validate_completed,
                        validate_resume=validate_resume,
                    )
                    for i in range(replicates)
                ]
                # A known system-wide resource failure is not evidence about
                # this posterior; preserve it and stop admitting more work.
                if any(
                    state.get("closure", {}).get("resource_exhaustion") is True
                    or state.get("closure", {}).get("errno") in _RESOURCE_ERRNOS
                    for state in states
                ):
                    stopping = True
                reserved = {s["replicate"] for s in states if s["reserved"]} | set(
                    owned
                )
                for state in states:
                    index = state["replicate"]
                    if stopping or len(reserved) >= workers:
                        break
                    if not state["eligible"] or index in attempted or index in reserved:
                        continue
                    issued = issue_attempt(
                        output,
                        index=index,
                        replicates=replicates,
                        identity=identity,
                        session=session,
                        command=command,
                    )
                    attempted.add(index)
                    issue = _read(issued)
                    directory = output / f"replicate-{index:03d}"
                    directory.mkdir(exist_ok=True)
                    log = (directory / f"attempt-{issue['attempt']}.log").open(
                        "ab", buffering=0
                    )
                    full_command = command + [
                        "--output",
                        str(output),
                        "--attempt",
                        str(issued),
                        "--index",
                        str(index),
                        "--replicates",
                        str(replicates),
                    ]
                    try:
                        process = _launch(full_command, log)
                    except OSError as exc:
                        log.close()
                        atomic_json(
                            _sidecar(issued, "exit"),
                            {
                                **{
                                    k: issue[k]
                                    for k in (
                                        "attempt",
                                        "replicate",
                                        "replicates",
                                        "identity",
                                    )
                                },
                                "returncode": None,
                                "exception": type(exc).__name__,
                                "message": str(exc),
                                "launch_failed": True,
                            },
                        )
                        emit(
                            "replica_launch_unresolved",
                            replicate=index,
                            message=str(exc),
                        )
                        stopping = True
                        break
                    owned[index] = (process, issued, issue, log)
                    reserved.add(index)
                    atomic_json(
                        _sidecar(issued, "launched"),
                        {
                            **{
                                k: issue[k]
                                for k in (
                                    "attempt",
                                    "replicate",
                                    "replicates",
                                    "identity",
                                )
                            },
                            "pid": process.pid,
                            "launched_unix": time.time(),
                        },
                    )
                    emit(
                        "replica_process_started",
                        replicate=index,
                        attempt=issue["attempt"],
                        pid=process.pid,
                    )
                # Re-read after launches to avoid publishing stale eligibility.
                states = [
                    inspect_replica(
                        output,
                        index=i,
                        replicates=replicates,
                        identity=identity,
                        validate_result=validate_completed,
                        validate_resume=validate_resume,
                    )
                    for i in range(replicates)
                ]
                records = [
                    s.get(
                        "result",
                        {k: v for k, v in s.items() if k not in ("result", "eligible")},
                    )
                    for s in states
                ]
                if progress is not None:
                    progress(records)
                orphan_active = any(
                    s["status"] == "active" and s["replicate"] not in owned
                    for s in states
                )
                remaining = any(
                    s["eligible"] and s["replicate"] not in attempted for s in states
                )
                reservations = sum(s["reserved"] for s in states)
                if (
                    not owned
                    and not orphan_active
                    and (stopping or not remaining or reservations >= workers)
                ):
                    break
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                stopping = True
                interrupted = True
                # Cancellation files reach orphan workers too, without unsafe PID signalling.
                for index in range(replicates):
                    for issued, issue in _issues(output, index, replicates, identity):
                        if (
                            _closure(output / f"replicate-{index:03d}", issued, issue)
                            is None
                        ):
                            atomic_json(
                                _sidecar(issued, "cancel"),
                                {
                                    "attempt": issue["attempt"],
                                    "requested_unix": time.time(),
                                },
                            )
                emit("sbc_cancellation_requested", session=session)
        for record in records:
            if record["status"] in ("qualified", "failed"):
                validate_result(output / f"replicate-{record['replicate']:03d}", record)
        atomic_json(
            session_path / "finished.json",
            {
                "finished_unix": time.time(),
                "interrupted": interrupted,
                "records": records,
            },
        )
        if interrupted:
            raise InterruptedError(
                "Calibration execution stopped; preserve committed evidence."
            )
        return records
    finally:
        # Normal completion has no owned processes. A second abrupt parent death
        # leaves their independently owned locks/attempts for reconciliation.
        for process, issued, issue, log in owned.values():
            log.close()


def _production_work(output, index, total, identity, cancel):
    # This import and all native verification occur AFTER replica ownership.
    from hmc_validation import _worker_replica

    return _worker_replica(
        output, index=index, replicates=total, identity=identity, cancel_check=cancel
    )


def execute_attempt(output, issued, *, index, replicates, work=None):
    output, issued = Path(output).resolve(), Path(issued).resolve()
    if not issued.is_relative_to(output / "executions"):
        raise ValueError("Attempt record must belong to this study.")
    issue = _read(issued)
    if issue["replicate"] != index or issue["replicates"] != replicates:
        raise ValueError("Worker index or fixed total differs from issued attempt.")
    _binding(output, replicates, issue["identity"])
    directory = output / f"replicate-{index:03d}"
    requested = False
    stage = "before_science"

    def signal_requested(signum, frame):
        nonlocal requested
        requested = True

    previous = {
        sig: signal.signal(sig, signal_requested)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def cancel(at_stage="safe_boundary"):
        nonlocal stage
        stage = at_stage
        if requested or _sidecar(issued, "cancel").exists():
            error = InterruptedError("Cooperative calibration interruption at " + stage)
            error.sbc_cooperative = True
            raise error

    common = {k: issue[k] for k in ("attempt", "replicate", "replicates", "identity")}
    try:
        with _replica_lock(directory):
            issues = _issues(output, index, replicates, issue["identity"])
            if not issues or issues[-1][1]["attempt"] != issue["attempt"]:
                raise ValueError(
                    "Worker is not the latest issued attempt for this index."
                )
            for prior_path, prior in issues[:-1]:
                closed = _closure(directory, prior_path, prior)
                if not (
                    closed
                    and closed.get("status") == "interrupted"
                    and closed.get("cooperative") is True
                ):
                    raise ValueError("A preceding attempt remains unresolved.")
            if (
                _closure_path(directory, issue).exists()
                or (directory / f"attempt-{issue['attempt']}.started.json").exists()
            ):
                raise ValueError("This issued attempt has already entered or closed.")
            atomic_json(
                directory / f"attempt-{issue['attempt']}.started.json",
                {**common, "pid": os.getpid(), "locked_unix": time.time()},
            )
            try:
                cancel("before_science")
                result = (work or _production_work)(
                    output, index, replicates, issue["identity"], cancel
                )
                if result.get("status") not in ("qualified", "failed"):
                    raise ValueError(
                        "Replica must return a qualified result or explicit scientific failure."
                    )
                result = {**result, **common}
                atomic_json(directory / "result.json", result)
                atomic_json(
                    _closure_path(directory, issue),
                    {**common, "status": result["status"], "closed_unix": time.time()},
                )
                return 0 if result["status"] == "qualified" else 2
            except (KeyboardInterrupt, InterruptedError) as exc:
                cooperative = getattr(exc, "sbc_cooperative", False) is True
                atomic_json(
                    _closure_path(directory, issue),
                    {
                        **common,
                        "status": "interrupted" if cooperative else "unresolved",
                        "cooperative": cooperative,
                        "stage": stage,
                        "boundary": getattr(exc, "sbc_boundary", None),
                        "exception": type(exc).__name__,
                        "message": str(exc),
                        "closed_unix": time.time(),
                    },
                )
                return 130 if cooperative else 1
            except Exception as exc:
                atomic_json(
                    _closure_path(directory, issue),
                    {
                        **common,
                        "status": "unresolved",
                        "exception": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "errno": getattr(exc, "errno", None),
                        "resource_exhaustion": (
                            isinstance(exc, MemoryError)
                            or getattr(exc, "errno", None) in _RESOURCE_ERRNOS
                        ),
                        "stage": stage,
                        "closed_unix": time.time(),
                    },
                )
                return 1
    except ReplicaBusy:
        return 75
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def worker_main(argv=None, *, work=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--replicates", required=True, type=int)
    args = parser.parse_args(argv)
    return execute_attempt(
        args.output,
        args.attempt,
        index=args.index,
        replicates=args.replicates,
        work=work,
    )


if __name__ == "__main__":
    raise SystemExit(worker_main())
