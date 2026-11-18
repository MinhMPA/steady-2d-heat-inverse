"""Independent full-field heat-posterior chains at durable physical boundaries."""

# standard-library imports
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic

# numerical imports
import numpy as np

# local imports
from hmc_blackjax import (
    HMCControls,
    BlackJAXChain,
    require_sampling_environment,
    encode_state,
    decode_state,
    encode_adaptation,
    decode_adaptation,
)
from posterior_store import ChainCheckpoint, TrajectoryRecord, _same


def _count(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class RunRequest:
    """Per-chain warmup/draw counts, master seed and frozen trajectory controls."""

    warmup: int
    draws: int
    chains: int
    seed: int
    controls: HMCControls

    def __post_init__(self):
        for name, lower in (("warmup", 0), ("draws", 1), ("chains", 1), ("seed", 0)):
            object.__setattr__(self, name, _count(getattr(self, name), name, lower))
        if self.seed >= 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        if not isinstance(self.controls, HMCControls):
            raise TypeError("controls must be HMCControls.")
        if self.warmup == 0 and (
            self.controls.adapt_step_size
            or not self.controls.initial_step_size_explicit
        ):
            raise ValueError(
                "Zero warmup requires explicitly disabled adaptation and an explicit step size."
            )

    def metadata(self):
        """Return immutable calculation controls and requested retained total."""
        return dict(
            warmup=self.warmup,
            total_draws=self.draws,
            chains=self.chains,
            seed=self.seed,
            controls=asdict(self.controls),
        )

    @classmethod
    def from_metadata(cls, metadata):
        """Restore the original controls including whether spacing was explicit."""
        fields = dict(metadata["controls"])
        explicit = fields.pop("initial_step_size_explicit")
        if not explicit:
            fields["initial_step_size"] = None
        return cls(
            metadata["warmup"],
            metadata["total_draws"],
            metadata["chains"],
            metadata["seed"],
            HMCControls(**fields),
        )


@dataclass(frozen=True)
class Progress:
    """A completed, committed physical trajectory; counts are per chain."""

    chain: int
    phase: str
    completed: int
    target: int
    elapsed_seconds: float
    output: Path


def _key_state(key):
    import jax

    return {
        "implementation": "threefry2x32",
        "key_data": np.array(jax.random.key_data(key), copy=True),
    }


def _restore_key(state):
    import jax

    if state["implementation"] != "threefry2x32":
        raise ValueError("Unsupported saved random implementation.")
    return jax.random.wrap_key_data(
        np.asarray(state["key_data"], dtype=np.uint32), impl="threefry2x32"
    )


def prepare_initial_states(states, request, *, disperse):
    """Fix every actual start and separate initialization/trajectory streams."""
    require_sampling_environment()
    import jax

    master = jax.random.key(request.seed, impl="threefry2x32")
    actual, streams = [], []
    for chain, state in enumerate(states):
        chain_key = jax.random.fold_in(master, chain)
        initial_key = jax.random.fold_in(chain_key, 0)
        trajectory_key = jax.random.fold_in(chain_key, 1)
        z = np.array(state.coordinates["z"], dtype=np.float64, copy=True)
        if disperse:
            z += request.controls.initial_spread * np.asarray(
                jax.random.normal(initial_key, z.shape, dtype=np.float64)
            )
        actual.append({"z": z})
        streams.append(
            {
                "initialization": _key_state(initial_key),
                "trajectories": _key_state(trajectory_key),
            }
        )
    return actual, streams


def _solves(posterior):
    return asdict(posterior.solve_counts)


def _difference(after, before):
    return {name: int(after[name] - before[name]) for name in before}


def _zero_solves():
    return dict(heat=0, adjoint=0, heat_iterations=0, adjoint_iterations=0)


class _EvaluationCache:
    """Retain only the preceding accepted field and most recent force probe."""

    def __init__(self, posterior, checkpoint=None):
        self.posterior, self.last = posterior, None
        self.retained = (
            None
            if checkpoint is None
            else (
                checkpoint.inferred_coordinates["z"].copy(),
                checkpoint.physical_state,
            )
        )

    def force(self, z):
        evaluation = self.posterior.evaluate(z)
        self.last = evaluation
        return np.float64(evaluation.potential), evaluation.gradient

    def physical(self, z):
        if self.retained is not None and np.array_equal(z, self.retained[0]):
            return self.retained[1]
        if self.last is None or not np.array_equal(z, self.last.z):
            self.force(z)
        fields = {
            "h": self.last.h.copy(),
            "sensor_temperature": self.last.sensor_temperature.copy(),
        }
        self.retained = (np.array(z, copy=True), fields)
        return fields


def run_chains(problem, *, initial_states, request: RunRequest, store, progress=None):
    """Finish independent conductivity chains, preserving exact continuation.

    Parameters
    ----------
    problem : HeatInverseProblem
        Frozen physical experiment; every chain owns a private heat workspace.
    initial_states : sequence or None
        Actual fixed coordinate mappings. None restores saved planned starts.
    request : RunRequest
        Original controls and current per-chain retained target.
    store : RunStore
        Exclusive writable calculation with fixed starts/random streams saved.
    progress : callable, optional
        Receives Progress only after a durable trajectory boundary.

    Returns
    -------
    None
        Exceptions preserve the preceding boundary and explicit abandoned cost.
    """
    require_sampling_environment()
    import jax

    store.validate_identity(problem.to_record())
    metadata = store.metadata
    if request.metadata() != {name: metadata[name] for name in request.metadata()}:
        # Tuple/list differences can arise only in externally constructed metadata;
        # the native lossless store preserves tuples exactly.
        raise ValueError("Run request differs from saved controls or target.")
    starts = metadata["initial_states"]
    if initial_states is not None:
        supplied = list(initial_states)
        if len(supplied) != len(starts) or any(
            not _same(a, b) for a, b in zip(supplied, starts)
        ):
            raise ValueError("Actual initial states differ from saved planned starts.")
    workspaces = {}
    active, phase, index, committed = None, "initialization", 0, False
    baseline = _zero_solves()
    started = monotonic()
    initialized = False
    try:
        try:
            store.checkpoint(0)
            initialized = True
        except KeyError:
            pass
        if not initialized:
            checkpoints = []
            for active in range(request.chains):
                posterior = problem._new_posterior(
                    prefix=f"hmc_chain_{active}_initial_"
                )
                workspaces[active] = posterior
                cache = _EvaluationCache(posterior)
                chain = BlackJAXChain(cache.force, request.controls)
                state = chain.initialize(starts[active]["z"])
                adaptation = chain.begin_warmup()
                counters = {
                    **_solves(posterior),
                    "trajectories": 0,
                    "accepted": 0,
                    "divergences": 0,
                }
                checkpoints.append(
                    ChainCheckpoint(
                        active,
                        "warmup" if request.warmup else "sample",
                        0,
                        0,
                        {"z": np.asarray(state.position).copy()},
                        cache.physical(np.asarray(state.position)),
                        encode_state(state),
                        metadata["random_streams"][active]["trajectories"],
                        encode_adaptation(adaptation),
                        request.controls.initial_step_size,
                        counters,
                    )
                )
            store.initialize_chains(checkpoints)
            initialized = True
        for active in range(request.chains):
            cp = store.checkpoint(active)
            if cp.draws_done >= request.draws:
                if active in workspaces:
                    workspaces.pop(active).close()
                continue
            posterior = workspaces.pop(active, None)
            if posterior is None:
                posterior = problem._new_posterior(
                    prefix=f"hmc_chain_{active}_continue_"
                )
            workspaces[active] = posterior
            cache = _EvaluationCache(posterior, cp)
            chain = BlackJAXChain(cache.force, request.controls)
            state, adaptation = decode_state(cp.hmc_state), decode_adaptation(
                cp.adaptation_state
            )
            while cp.warmup_done < request.warmup or cp.draws_done < request.draws:
                phase = "warmup" if cp.warmup_done < request.warmup else "sample"
                index = cp.warmup_done if phase == "warmup" else cp.draws_done
                committed = False
                baseline = _solves(posterior)
                trajectory_started = monotonic()
                next_key, duration_key, transition_key = jax.random.split(
                    _restore_key(cp.rng_state), 3
                )
                duration = float(
                    jax.random.uniform(
                        duration_key,
                        (),
                        minval=request.controls.trajectory_interval[0],
                        maxval=request.controls.trajectory_interval[1],
                        dtype=np.float64,
                    )
                )
                step_size = cp.step_size
                num_steps = max(1, int(np.floor(duration / step_size + 0.5)))
                transition = chain.transition(
                    state, key=transition_key, step_size=step_size, num_steps=num_steps
                )
                state = transition.state
                z = np.asarray(state.position).copy()
                physical = cache.physical(z)
                spacing = step_size
                if phase == "warmup":
                    adaptation = chain.adapt(
                        adaptation, transition.acceptance_probability
                    )
                    spacing = (
                        chain.final_step_size(adaptation)
                        if index + 1 == request.warmup
                        else chain.current_step_size(adaptation)
                    )
                counters = dict(cp.counters)
                for name, increment in _difference(
                    _solves(posterior), baseline
                ).items():
                    counters[name] += increment
                counters["trajectories"] += 1
                counters["accepted"] += int(transition.accepted)
                counters["divergences"] += int(transition.divergent)
                statistics = {
                    name: value
                    for name, value in asdict(transition).items()
                    if name != "state"
                }
                statistics.update(
                    step_size=step_size,
                    duration=duration,
                    actual_duration=num_steps * step_size,
                    trajectory_seconds=monotonic() - trajectory_started,
                    solve_counts=_difference(_solves(posterior), baseline),
                )
                cp = replace(
                    cp,
                    phase=phase,
                    warmup_done=cp.warmup_done + int(phase == "warmup"),
                    draws_done=cp.draws_done + int(phase == "sample"),
                    inferred_coordinates={"z": z},
                    physical_state=physical,
                    hmc_state=encode_state(state),
                    rng_state=_key_state(next_key),
                    adaptation_state=encode_adaptation(adaptation),
                    step_size=spacing,
                    counters=counters,
                )
                record = TrajectoryRecord(
                    active,
                    phase,
                    index,
                    cp.inferred_coordinates,
                    physical["h"],
                    physical["sensor_temperature"],
                    statistics,
                )
                store.commit_trajectory(record, cp)
                committed = True
                if progress is not None:
                    progress(
                        Progress(
                            active,
                            phase,
                            index + 1,
                            request.warmup if phase == "warmup" else request.draws,
                            monotonic() - started,
                            store.path,
                        )
                    )
            workspaces.pop(active).close()
    except BaseException as exc:
        # The database may already contain this boundary even if an interrupt
        # arrived before the save call returned (including advisory status).
        # Count durable physical work once, using the authoritative checkpoint.
        try:
            durable = store.checkpoint(0 if active is None else active)
        except KeyError:
            durable = None
        if durable is not None:
            if not initialized:
                initialized, committed = True, True
            elif phase in ("warmup", "sample"):
                completed = (
                    durable.warmup_done if phase == "warmup" else durable.draws_done
                )
                committed = completed > index
        abandoned = _zero_solves()
        if not initialized:
            for posterior in workspaces.values():
                for name, value in _solves(posterior).items():
                    abandoned[name] += value
        elif not committed and active in workspaces:
            abandoned = _difference(_solves(workspaces[active]), baseline)
        evidence = dict(
            chain=active,
            phase=phase,
            trajectory=index,
            output=str(store.path),
            exception=type(exc).__name__,
            message=str(exc),
            committed=committed,
            aborted_work=abandoned,
        )
        for name in ("stage", "position", "reason", "residual_norm", "iterations"):
            if hasattr(exc, name):
                evidence[name] = getattr(exc, name)
        store.write_failure(evidence)
        if hasattr(exc, "add_note"):
            exc.add_note(f"Partial posterior calculation saved at {store.path}")
        raise
    finally:
        for posterior in workspaces.values():
            posterior.close()
