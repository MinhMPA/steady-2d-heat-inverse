"""Eager CPU connection to the supplied finite-element posterior and adjoint."""

# standard-library imports
from dataclasses import dataclass, field
import importlib
from math import pi
from numbers import Integral
import platform
import sys

# numerical imports
import numpy as np

_SAMPLING_VERSIONS = {"jax": "0.11.2", "jaxlib": "0.11.2", "blackjax": "1.7.1"}
_EAGER_MESSAGE = (
    "The finite-element posterior requires eager CPU evaluation; "
    "jit, vmap, pmap and other traced evaluations are unsupported. "
    "Use jax.value_and_grad under jax.disable_jit()."
)


def require_sampling_environment():
    r"""Validate the qualified CPU, float64 sampling calculation.

    Returns
    -------
    dict
        Actual Python and dependency versions, CPU backend, and verified dtype.
        Sampling requires Python 3.12+; legacy reconstruction remains available
        on the project's existing Python versions. Optional sampling imports
        occur only here, and the caller's JAX configuration is never changed.
    """
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Full-field sampling requires Python 3.12 or newer. "
            "Legacy reconstruction remains supported on Python 3.11."
        )
    try:
        modules = {name: importlib.import_module(name) for name in _SAMPLING_VERSIONS}
    except ImportError as exc:
        raise ImportError(
            "Full-field sampling requires the optional [sampling] dependencies. "
            "Install the sampling extra with constraints/hmc-cpu-py312.txt."
        ) from exc
    versions = {name: module.__version__ for name, module in modules.items()}
    for name, expected in _SAMPLING_VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(
                f"Sampling qualification requires {name}=={expected}; "
                f"found {versions[name]}. Use constraints/hmc-cpu-py312.txt."
            )
    jax = modules["jax"]
    jnp = importlib.import_module("jax.numpy")
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Sampling requires real float64 JAX arrays. Set JAX_ENABLE_X64=true "
            "before starting Python; the sampling route does not change it."
        )
    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise RuntimeError(
            "Sampling requires the eager CPU backend. Set JAX_PLATFORMS=cpu "
            "before starting Python."
        )
    probe = jnp.asarray(np.array([1.0], dtype=np.float64))
    if probe.dtype != jnp.float64 or any(
        device.platform != "cpu" for device in probe.devices()
    ):
        raise RuntimeError("Sampling requires real CPU float64 JAX arrays.")
    return {
        "python": platform.python_version(),
        **versions,
        "backend": "cpu",
        "float_dtype": str(probe.dtype),
    }


def make_logdensity(value_and_grad):
    r"""Expose :math:`-U` and its supplied derivative :math:`-\nabla U` to JAX.

    Parameters
    ----------
    value_and_grad : callable
        Receives one copied NumPy float64 position vector and returns its scalar
        potential and NumPy float64 gradient together. Its heat-solve and
        nonfinite-proposal exceptions pass through unchanged.

    Returns
    -------
    callable
        The eager CPU log probability. A combined ``jax.value_and_grad`` request
        evaluates the supplied function exactly once and retains the supplied
        force for reverse differentiation. Use it under ``jax.disable_jit()``;
        traced compilation, batching, and parallel mapping are unsupported.
    """
    if not callable(value_and_grad):
        raise TypeError("value_and_grad must supply the posterior potential and force.")
    require_sampling_environment()
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")

    def evaluate(z):
        # The custom reverse rule receives a concrete primal for an eager
        # gradient. Reject other transformations before touching NumPy/PETSc.
        if isinstance(z, jax.core.Tracer):
            raise RuntimeError(_EAGER_MESSAGE)
        if not jax.config.x64_enabled or np.dtype(z.dtype) != np.dtype(np.float64):
            raise RuntimeError(
                "The posterior position must use real float64 JAX arrays."
            )
        if isinstance(z, jax.Array) and any(
            device.platform != "cpu" for device in z.devices()
        ):
            raise RuntimeError(_EAGER_MESSAGE)
        position = np.array(z, copy=True)
        if position.ndim != 1:
            raise ValueError("The full-field posterior position must be one vector.")
        # Keep this in the ordinary Python calling sequence: PDE failures are
        # fatal diagnostics and nonfinite proposals retain their separate type.
        potential, gradient = value_and_grad(position)
        potential, gradient = np.asarray(potential), np.asarray(gradient)
        if potential.shape != () or gradient.shape != position.shape:
            raise ValueError(
                "The supplied potential and force must be scalar and vector."
            )
        if potential.dtype != np.float64 or gradient.dtype != np.float64:
            raise RuntimeError(
                "The supplied posterior potential and force must be float64."
            )
        return jnp.asarray(-potential), jnp.asarray(-gradient)

    @jax.custom_vjp
    def logdensity(z):
        return evaluate(z)[0]

    def forward(z):
        value, force = evaluate(z)
        return value, force

    def reverse(force, weight):
        return (weight * force,)

    logdensity.defvjp(forward, reverse)
    return logdensity


# These declarations intentionally need no optional sampling imports.
def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).shape != ():
        raise ValueError(f"{name} must be one finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be one finite real number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be one finite real number.")
    return result


def _interval(value, name):
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two finite real numbers.") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must contain two finite real numbers.")
    return tuple(_finite_scalar(item, name) for item in items)


@dataclass(frozen=True)
class HMCControls:
    r"""Recorded controls for the Gaussian-prior Hamiltonian calculation.

    Parameters
    ----------
    initial_step_size, target_acceptance : float
        Initial integration spacing and desired Metropolis acceptance in warmup.
    trajectory_interval, step_size_bounds : pair of float
        Duration interval and allowed integration spacing. The duration and
        spacing constraints keep rounded trajectories away from prior resonance.
    adapt_step_size : bool
        Tune only integration spacing during warmup; momentum covariance is fixed.
    divergence_threshold, initial_spread : float
        Absolute energy-error diagnostic threshold and initial whitened spread.
        These defaults remain provisional pending physical qualification.
    """

    initial_step_size: float | None = None
    initial_step_size_explicit: bool = field(init=False)
    target_acceptance: float = 0.8
    trajectory_interval: tuple = (pi / 4, 3 * pi / 4)
    adapt_step_size: bool = True
    step_size_bounds: tuple = (1e-6, pi / 8)
    divergence_threshold: float = 1000.0
    initial_spread: float = 0.3

    def __post_init__(self):
        object.__setattr__(
            self, "initial_step_size_explicit", self.initial_step_size is not None
        )
        if self.initial_step_size is None:
            object.__setattr__(self, "initial_step_size", 0.1)
        for name in (
            "initial_step_size",
            "target_acceptance",
            "divergence_threshold",
            "initial_spread",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        for name in ("trajectory_interval", "step_size_bounds"):
            object.__setattr__(self, name, _interval(getattr(self, name), name))
        if not isinstance(self.adapt_step_size, (bool, np.bool_)):
            raise TypeError("adapt_step_size must be boolean.")
        object.__setattr__(self, "adapt_step_size", bool(self.adapt_step_size))
        tmin, tmax = self.trajectory_interval
        lower, upper = self.step_size_bounds
        if not 0 < tmin < tmax < pi:
            raise ValueError("Require 0 < Tmin < Tmax < pi for the prior rotation.")
        if not 0 < lower <= upper <= min(tmin / 2, pi - tmax):
            raise ValueError(
                "Step-size bounds must be positive and avoid prior resonance."
            )
        if not lower <= self.initial_step_size <= upper:
            raise ValueError("Initial step size must lie within its recorded bounds.")
        if not 0 < self.target_acceptance < 1:
            raise ValueError(
                "Target acceptance must lie strictly between zero and one."
            )
        if self.divergence_threshold <= 0 or self.initial_spread < 0:
            raise ValueError(
                "Divergence threshold must be positive and spread nonnegative."
            )


@dataclass(frozen=True)
class Transition:
    """One retained or repeated state and its Hamiltonian diagnostics.

    Energies use ``final - initial`` for ``delta_energy``. Missing energies and
    the ordinary no-failure reason are ``None``. A finite large energy error is
    a diagnostic; BlackJAX still makes the ordinary Metropolis decision.
    """

    state: object
    acceptance_probability: float
    accepted: bool
    divergent: bool
    delta_energy: object
    initial_energy: object
    final_energy: object
    num_steps: int
    failure_reason: object


def _require_finite(stage, position, *values):
    if any(not np.isfinite(np.asarray(value)).all() for value in values):
        from conductivity_posterior import NonfiniteProposal

        raise NonfiniteProposal(stage, np.asarray(position))


def gaussian_split_integrator(logdensity_fn, kinetic_energy):
    r"""Return likelihood half-kicks around an exact unit Gaussian rotation.

    Parameters
    ----------
    logdensity_fn : callable
        Full log density ``-Phi(z) - 0.5*z@z`` with its supplied adjoint.
    kinetic_energy : callable
        BlackJAX kinetic energy for the fixed identity momentum covariance.

    Returns
    -------
    callable
        A BlackJAX ``IntegratorState, step_size -> IntegratorState`` map.
        The kick uses ``grad(Phi) = -logdensity_grad - z``. Only the
        likelihood is approximated; the Gaussian prior and kinetic energy
        evolve together by an exact phase-space rotation.
    """
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    state_type = importlib.import_module("blackjax.mcmc.integrators").IntegratorState
    value_and_grad = jax.value_and_grad(logdensity_fn)

    def step(state, step_size):
        z, p, value, grad = state
        _require_finite("initial Hamiltonian", z, z, p, value, grad, kinetic_energy(p))
        # The prior force has already been included in logdensity_grad.
        p = p + (0.5 * step_size) * (grad + z)
        cosine, sine = jnp.cos(step_size), jnp.sin(step_size)
        new_z = cosine * z + sine * p
        new_p = -sine * z + cosine * p
        _require_finite("Gaussian rotation", new_z, new_z, new_p)
        value, grad = value_and_grad(new_z)
        _require_finite("posterior value or force", new_z, value, grad)
        new_p = new_p + (0.5 * step_size) * (grad + new_z)
        _require_finite(
            "final Hamiltonian",
            new_z,
            new_p,
            kinetic_energy(new_p),
            -value + kinetic_energy(new_p),
        )
        return state_type(new_z, new_p, value, grad)

    return step


def _copy_leaf(value):
    # Preserve scalar-vs-array identity, dtype and shape in named storage fields.
    # A JAX scalar is an array of shape (), never a Python float sentinel.
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        return np.array(value, copy=True)
    if isinstance(value, (float, int, bool, str, type(None), np.generic)):
        return value
    raise TypeError(f"Unsupported library state field type: {type(value).__name__}.")


def _encode_named(state):
    return {name: _copy_leaf(getattr(state, name)) for name in state._fields}


def _decode_named(fields, state_type):
    if not isinstance(fields, dict) or set(fields) != set(state_type._fields):
        raise ValueError(
            "Saved fields must contain every named library state field exactly."
        )
    jnp = importlib.import_module("jax.numpy")
    return state_type(
        **{
            name: (
                jnp.asarray(np.array(value, copy=True))
                if isinstance(value, np.ndarray)
                else _copy_leaf(value)
            )
            for name, value in fields.items()
        }
    )


def encode_state(state):
    """Copy every named BlackJAX HMC field to lossless NumPy/scalar storage."""
    return _encode_named(state)


def decode_state(fields):
    """Restore all named HMC fields, preserving array shape and dtype."""
    require_sampling_environment()
    state_type = importlib.import_module("blackjax.mcmc.hmc").HMCState
    return _decode_named(fields, state_type)


def encode_adaptation(state):
    """Copy every pinned dual-averaging field, including its iteration state."""
    return _encode_named(state)


def decode_adaptation(fields):
    """Restore all named dual-averaging fields without dropping library fields."""
    require_sampling_environment()
    state_type = importlib.import_module(
        "blackjax.adaptation.step_size"
    ).DualAveragingAdaptationState
    return _decode_named(fields, state_type)


def _copy_state(state):
    # Library states are immutable containers, but copies also isolate future
    # host storage conversions from the retained physical coordinates.
    jnp = importlib.import_module("jax.numpy")
    return type(state)(*(jnp.array(value, copy=True) for value in state))


def _hamiltonian(logdensity, momentum):
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(
            -np.asarray(logdensity)
            + 0.5 * np.dot(np.asarray(momentum), np.asarray(momentum))
        )
    return value if np.isfinite(value) else None


class BlackJAXChain:
    r"""Eager CPU HMC for all whitened conductivity coordinates.

    Parameters
    ----------
    value_and_grad : callable
        Full potential and supplied adjoint derivative at a copied float64 vector.
    controls : HMCControls
        Frozen physical-trajectory and step-size-only warmup settings.

    Notes
    -----
    BlackJAX refreshes momenta, constructs the endpoint proposal, and chooses
    the retained state by ordinary Metropolis acceptance. The sole exceptional
    rejection is nonfinite arithmetic. An unconverged heat solve propagates.
    """

    def __init__(self, value_and_grad, controls):
        if not isinstance(controls, HMCControls):
            raise TypeError("controls must be frozen HMCControls.")
        self.controls = controls
        self.logdensity = make_logdensity(value_and_grad)
        self._jax = importlib.import_module("jax")
        self._jnp = importlib.import_module("jax.numpy")
        self._hmc = importlib.import_module("blackjax.mcmc.hmc")
        adaptation = importlib.import_module("blackjax.adaptation.step_size")
        self._adapt_init, self._adapt_update, self._adapt_final = (
            adaptation.dual_averaging_adaptation(
                controls.target_acceptance, t0=10, gamma=0.05, kappa=0.75
            )
        )

    def initialize(self, z):
        """Evaluate one finite initial potential and supplied adjoint force.

        Parameters
        ----------
        z : array-like
            All whitened conductivity coordinates, as a finite vector.

        Returns
        -------
        HMCState
            Copied position, full log density and its supplied derivative.
        """
        z = np.array(z, dtype=np.float64, copy=True)
        if z.ndim != 1 or z.size == 0:
            raise ValueError("Initial whitened conductivity must be a nonempty vector.")
        _require_finite("initial position", z, z)
        with self._jax.disable_jit():
            state = self._hmc.init(self._jnp.asarray(z), self.logdensity)
        _require_finite("initial posterior", z, state.logdensity, state.logdensity_grad)
        return _copy_state(state)

    def transition(self, state, *, key, step_size, num_steps):
        """Advance one trajectory, retaining all repeated-state outcomes.

        Parameters
        ----------
        state : HMCState
            Previous whitened field with its full log density and derivative.
        key : JAX random key
            Preallocated transition key for momentum and Metropolis randomness.
        step_size : float
            Integration spacing within the recorded bounds.
        num_steps : int
            Positive number of split Hamiltonian steps.

        Returns
        -------
        Transition
            Copied new or repeated state, acceptance and energy diagnostics.
            An unconverged heat or adjoint solve propagates to the caller.
        """
        step_size = self._bounded_step_size(step_size, cap_upper=False)
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, Integral)
            or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer.")
        num_steps = int(num_steps)
        from conductivity_posterior import NonfiniteProposal

        initial_energy = None
        # This observer sees the actual BlackJAX momentum once; it neither
        # draws momentum nor constructs a second proposal/acceptance decision.
        observed = False

        def observed_integrator(logdensity_fn, kinetic_energy):
            step = gaussian_split_integrator(logdensity_fn, kinetic_energy)

            def observe(integrator_state, eps):
                nonlocal initial_energy, observed
                if not observed:
                    initial_energy = _hamiltonian(
                        integrator_state.logdensity, integrator_state.momentum
                    )
                    observed = True
                return step(integrator_state, eps)

            return observe

        kernel = self._hmc.build_kernel(
            integrator=observed_integrator,
            divergence_threshold=self.controls.divergence_threshold,
        )
        try:
            with self._jax.disable_jit():
                new_state, info = kernel(
                    key,
                    state,
                    self.logdensity,
                    step_size,
                    self._jnp.ones_like(state.position),
                    num_steps,
                )
        except NonfiniteProposal as exc:
            return Transition(
                _copy_state(state),
                0.0,
                False,
                True,
                None,
                initial_energy,
                None,
                num_steps,
                f"nonfinite {exc.stage}",
            )
        final_energy = float(info.energy)
        if not np.isfinite(final_energy):
            final_energy = None
        with np.errstate(over="ignore", invalid="ignore"):
            delta = (
                final_energy - initial_energy
                if final_energy is not None and initial_energy is not None
                else None
            )
        delta = delta if delta is not None and np.isfinite(delta) else None
        probability = float(info.acceptance_rate)
        if (
            delta is None
            or not np.isfinite(probability)
            or any(not np.isfinite(np.asarray(value)).all() for value in info.proposal)
        ):
            return Transition(
                _copy_state(state),
                0.0,
                False,
                True,
                delta,
                initial_energy,
                final_energy,
                num_steps,
                "nonfinite Hamiltonian or proposal",
            )
        return Transition(
            _copy_state(new_state),
            probability,
            bool(info.is_accepted),
            abs(delta) > self.controls.divergence_threshold,
            delta,
            initial_energy,
            final_energy,
            num_steps,
            None,
        )

    def begin_warmup(self):
        """Initialize the recorded step-size warmup.

        Returns
        -------
        DualAveragingAdaptationState
            Every named field of the pinned BlackJAX warmup state.
        """
        with self._jax.disable_jit():
            return self._adapt_init(self.controls.initial_step_size)

    def _bounded_step_size(self, proposed, *, cap_upper=True):
        lower, upper = self.controls.step_size_bounds
        proposed = _finite_scalar(proposed, "step_size")

        def roundoff(bound):
            # A logarithm's absolute rounding error propagates to relative
            # spacing error on exponentiation. Keep only a few such errors,
            # with a few-ULP floor for normal values. A subnormal ULP can be
            # a material fraction of its bound, so it gets no absolute floor.
            precision = np.finfo(np.float64)
            tolerance = 4 * precision.eps * max(1.0, abs(np.log(bound))) * bound
            if bound >= precision.tiny:
                tolerance = max(tolerance, 4 * np.spacing(bound))
            return tolerance

        # Reject zero/negative spacings before any boundary normalization.
        if proposed > 0 and proposed < lower and lower - proposed <= roundoff(lower):
            proposed = lower
        if proposed <= 0 or proposed < lower:
            raise ValueError(
                f"Required step size {proposed!r} is below the lower bound {lower!r}; "
                "inspect warmup acceptance and physical force scales."
            )
        if proposed > upper:
            if cap_upper or proposed - upper <= roundoff(upper):
                return upper
            raise ValueError("Trajectory step size must lie within recorded bounds.")
        return proposed

    def current_step_size(self, adaptation):
        """Return the physical spacing for the next warmup trajectory.

        Parameters
        ----------
        adaptation : DualAveragingAdaptationState
            Complete current warmup state, containing the proposed log spacing.

        Returns
        -------
        float
            Spacing within the exact recorded bounds, with logarithmic
            roundoff normalized to the violated boundary. Manual controls
            return the explicitly chosen spacing. Callers record this value
            rather than interpreting the library's logarithmic field.
        """
        if not self.controls.adapt_step_size:
            return self.controls.initial_step_size
        with self._jax.disable_jit():
            _finite_scalar(adaptation.log_step_size, "proposed log step size")
            capped_log_size = self._jnp.minimum(
                adaptation.log_step_size, np.log(self.controls.step_size_bounds[1])
            )
            return self._bounded_step_size(self._jnp.exp(capped_log_size))

    def adapt(self, adaptation, acceptance_probability):
        """Update warmup and cap the next Hamiltonian integration spacing.

        Parameters
        ----------
        adaptation : DualAveragingAdaptationState
            Complete current step-size warmup state.
        acceptance_probability : float
            Metropolis acceptance probability in the closed interval [0, 1].

        Returns
        -------
        DualAveragingAdaptationState
            Complete updated state, or the supplied state under manual controls.
            A required spacing below the lower bound raises with evidence.
        """
        probability = _finite_scalar(acceptance_probability, "acceptance_probability")
        if not 0 <= probability <= 1:
            raise ValueError("Acceptance probability must lie between zero and one.")
        if not self.controls.adapt_step_size:
            return adaptation
        with self._jax.disable_jit():
            updated = self._adapt_update(adaptation, probability)
            self.current_step_size(updated)
            # Cap in logarithmic coordinates before exponentiation: a finite
            # warmup proposal may have an unrepresentably large exponential.
            capped_log_size = self._jnp.minimum(
                updated.log_step_size, np.log(self.controls.step_size_bounds[1])
            )
            return updated._replace(log_step_size=capped_log_size)

    def final_step_size(self, adaptation):
        """Fix the averaged spacing for the retained conductivity draws.

        Parameters
        ----------
        adaptation : DualAveragingAdaptationState
            Complete final warmup state, including the averaged log spacing.

        Returns
        -------
        float
            Bounded averaged spacing, or the explicitly chosen manual spacing.
        """
        if not self.controls.adapt_step_size:
            return self.controls.initial_step_size
        with self._jax.disable_jit():
            _finite_scalar(adaptation.log_step_size_avg, "averaged log step size")
            capped = adaptation._replace(
                log_step_size_avg=self._jnp.minimum(
                    adaptation.log_step_size_avg,
                    np.log(self.controls.step_size_bounds[1]),
                )
            )
            return self._bounded_step_size(self._adapt_final(capped))
