"""The eager JAX probability and force retain the finite-element posterior."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from hmc_blackjax import make_logdensity, require_sampling_environment
import jax
import jax.numpy as jnp
from conductivity_posterior import HeatSolveFailure, NonfiniteProposal
from ._hmc_helpers import make_posterior


def quadratic(z):
    return 0.5 * z @ z, z.copy()


def run_child(code, **environment):
    env = dict(os.environ, **environment)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_environment_is_real_cpu_float64_and_records_qualified_versions():
    before = (jax.config.x64_enabled, jax.default_backend())
    record = require_sampling_environment()
    assert record["jax"] == "0.11.2"
    assert record["jaxlib"] == "0.11.2"
    assert record["blackjax"] == "1.7.1"
    assert record["backend"] == "cpu"
    assert record["float_dtype"] == "float64"
    assert jnp.asarray(1.0, dtype=jnp.float64).dtype == jnp.float64
    assert all(device.platform == "cpu" for device in jax.devices())
    assert before == (jax.config.x64_enabled, jax.default_backend())


def test_connected_value_and_force_equal_direct_adjoint():
    with make_posterior() as posterior:
        z = np.linspace(-0.1, 0.1, posterior.prior.dimension)
        direct = posterior.evaluate(z)
        calls = []

        def value_and_grad(x):
            assert isinstance(x, np.ndarray)
            assert x.dtype == np.float64
            calls.append(x.copy())
            ev = posterior.evaluate(x)
            return ev.potential, ev.gradient

        logdensity = make_logdensity(value_and_grad)
        with jax.disable_jit():
            value, force = jax.value_and_grad(logdensity)(jnp.asarray(z))
            assert len(calls) == 1
            assert posterior.solve_counts.heat == 2
            assert posterior.solve_counts.adjoint == 2
            other = jax.value_and_grad(logdensity)(jnp.asarray(-z))
            again = jax.value_and_grad(logdensity)(jnp.asarray(z))
        assert len(calls) == 3
        assert value.dtype == force.dtype == jnp.float64
        assert float(value) == -direct.potential
        assert np.array_equal(np.asarray(force), -direct.gradient)
        assert float(again[0]) == float(value)
        assert np.array_equal(np.asarray(again[1]), np.asarray(force))
        assert np.isfinite(np.asarray(other[1])).all()


def test_one_evaluation_supplies_both_value_and_force_and_reverse_weight():
    calls = []

    def supplied(z):
        calls.append(z.copy())
        return quadratic(z)

    logdensity = make_logdensity(supplied)
    z = jnp.asarray([0.25, -0.5, 2.0], dtype=jnp.float64)
    with jax.disable_jit():
        value, force = jax.value_and_grad(logdensity)(z)
        assert len(calls) == 1
        assert float(value) == -0.5 * float(z @ z)
        assert np.array_equal(np.asarray(force), -np.asarray(z))
        weighted = jax.grad(lambda x: 3.0 * logdensity(x))(z)
        assert len(calls) == 2
        assert np.array_equal(np.asarray(weighted), -3.0 * np.asarray(z))
        assert float(logdensity(z)) == float(value)
        assert len(calls) == 3


@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize("failure_type", [HeatSolveFailure, NonfiniteProposal])
def test_supplied_exceptions_keep_identity_and_diagnostics(failure_type, later):
    z = np.array([0.25, -0.5])
    if failure_type is HeatSolveFailure:
        failure = HeatSolveFailure("adjoint", z, -3, 0.125, 7)
    else:
        failure = NonfiniteProposal("force", z)
    calls = 0

    def supplied(x):
        nonlocal calls
        calls += 1
        if calls == (2 if later else 1):
            raise failure
        return quadratic(x)

    logdensity = make_logdensity(supplied)
    with jax.disable_jit():
        if later:
            jax.value_and_grad(logdensity)(jnp.asarray(z))
        with pytest.raises(failure_type) as exc:
            jax.value_and_grad(logdensity)(jnp.asarray(z))
        assert exc.value is failure
        assert exc.value.stage in ("adjoint", "force")
        assert np.array_equal(exc.value.position, z)
        if failure_type is NonfiniteProposal:
            value, force = jax.value_and_grad(logdensity)(jnp.asarray(z))
            assert np.isfinite(value) and np.isfinite(np.asarray(force)).all()


@pytest.mark.parametrize("stage", ["heat", "adjoint"])
@pytest.mark.parametrize("later", [False, True])
def test_real_unconverged_solve_stops_eager_evaluation(stage, later):
    with make_posterior() as posterior:
        z = np.zeros(posterior.prior.dimension)

        def supplied(x):
            ev = posterior.evaluate(x)
            return ev.potential, ev.gradient

        logdensity = make_logdensity(supplied)
        with jax.disable_jit():
            if later:
                jax.value_and_grad(logdensity)(jnp.asarray(z))
            solver = (
                posterior._forward.problem.solver
                if stage == "heat"
                else posterior._adjoint_solver
            )
            solver.setTolerances(rtol=1e-14, atol=0, max_it=1)
            solver.getPC().setType("none")
            with pytest.raises(HeatSolveFailure) as exc:
                jax.value_and_grad(logdensity)(jnp.asarray(z))
        assert exc.value.stage == stage
        assert exc.value.reason < 0
        assert exc.value.iterations <= 1
        assert np.array_equal(exc.value.position, z)


def test_real_nonfinite_proposal_remains_recoverable_after_valid_force():
    with make_posterior() as posterior:
        z = np.zeros(posterior.prior.dimension)

        def supplied(x):
            ev = posterior.evaluate(x)
            return ev.potential, ev.gradient

        logdensity = make_logdensity(supplied)
        with jax.disable_jit():
            initial = jax.value_and_grad(logdensity)(jnp.asarray(z))
            counts = posterior.solve_counts
            with pytest.raises(NonfiniteProposal) as exc:
                jax.value_and_grad(logdensity)(jnp.full(z.shape, jnp.nan))
            assert exc.value.stage == "position"
            assert posterior.solve_counts == counts
            again = jax.value_and_grad(logdensity)(jnp.asarray(z))
        assert float(initial[0]) == float(again[0])
        assert np.array_equal(np.asarray(initial[1]), np.asarray(again[1]))


@pytest.mark.parametrize("transform", ["jit", "vmap", "pmap"])
def test_unsupported_transform_fails_before_a_posterior_evaluation(transform):
    calls = []

    def supplied(z):
        calls.append(z)
        return quadratic(z)

    logdensity = make_logdensity(supplied)
    z = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    with jax.disable_jit(False):
        transformed = getattr(jax, transform)(logdensity)
        with pytest.raises(RuntimeError, match="eager.*CPU|CPU.*eager"):
            transformed(z if transform == "jit" else z[None, :])
    assert not calls


def test_float32_position_is_rejected_before_evaluation():
    calls = []
    logdensity = make_logdensity(lambda z: calls.append(z) or quadratic(z))
    with jax.disable_jit(), pytest.raises(RuntimeError, match="float64"):
        jax.value_and_grad(logdensity)(jnp.asarray([0.1], dtype=jnp.float32))
    assert not calls


def test_float32_environment_fails_without_enabling_global_precision():
    run_child(
        """
        import jax
        from hmc_blackjax import require_sampling_environment
        assert not jax.config.x64_enabled
        try:
            require_sampling_environment()
        except RuntimeError as exc:
            assert "float64" in str(exc)
        else:
            raise AssertionError("Float32 environment was accepted.")
        assert not jax.config.x64_enabled
    """,
        JAX_ENABLE_X64="false",
        JAX_PLATFORMS="cpu",
    )


def test_import_is_lazy_and_legacy_python_gets_sampling_only_message():
    run_child("""
        import sys
        import hmc_blackjax
        assert "jax" not in sys.modules
        assert "blackjax" not in sys.modules
        sys.version_info = (3, 11, 0)
        try:
            hmc_blackjax.require_sampling_environment()
        except RuntimeError as exc:
            assert "sampling" in str(exc).lower()
            assert "3.12" in str(exc)
        else:
            raise AssertionError("Python 3.11 sampling was accepted.")
        assert "jax" not in sys.modules
        assert "blackjax" not in sys.modules
    """)


def test_missing_optional_dependencies_have_install_guidance():
    run_child("""
        import importlib.abc
        import sys
        from hmc_blackjax import require_sampling_environment
        class MissingSampling(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in {"jax", "jaxlib", "blackjax"}:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, MissingSampling())
        try:
            require_sampling_environment()
        except ImportError as exc:
            assert "sampling" in str(exc)
        else:
            raise AssertionError("Missing optional sampling was accepted.")
    """)
