"""Eager CPU connection to the supplied finite-element posterior and adjoint."""

# standard-library imports
import importlib
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
