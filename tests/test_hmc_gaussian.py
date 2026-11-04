"""Four seeded chains recover exact Gaussian means and covariances with MCSEs."""

import os

import pytest

if os.environ.get("S2DHI_HMC_QUALIFICATION") != "1":
    pytest.skip(
        "Set S2DHI_HMC_QUALIFICATION=1 to require eager CPU float64 HMC qualification.",
        allow_module_level=True,
    )

from hmc_blackjax import require_sampling_environment

require_sampling_environment()

import time
import numpy as np
import jax
from hmc_blackjax import BlackJAXChain, HMCControls
from ._hmc_helpers import gaussian_value_and_grad

SEEDS = (14101, 14102, 14103, 14104)
BURN = 100
DRAWS = 1200
BATCH = 100
MEAN = np.array([0.3, -0.4, 0.2])
TARGETS = [
    np.diag([0.7, 1.1, 1.4]),
    np.array([[0.9, 0.24, -0.15], [0.24, 0.7, 0.1], [-0.15, 0.1, 1.2]]),
]


def batch_estimates(draws, mean):
    # Each chain contributes independent contiguous batch estimates. Both first
    # and second moments use the exact target centering, with no fitted offset.
    centered = draws - mean
    statistics = np.concatenate(
        [
            draws,
            (centered[..., :, None] * centered[..., None, :]).reshape(
                *draws.shape[:2], -1
            ),
        ],
        axis=-1,
    )
    batches = (
        statistics.reshape(draws.shape[0], draws.shape[1] // BATCH, BATCH, -1)
        .mean(axis=2)
        .reshape(-1, statistics.shape[-1])
    )
    return statistics.mean(axis=(0, 1)), batches.std(axis=0, ddof=1) / np.sqrt(
        len(batches)
    )


@pytest.mark.parametrize(
    "mean,covariance",
    [(np.zeros(3), np.eye(3)), (MEAN, TARGETS[0]), (MEAN, TARGETS[1])],
    ids=["prior-only", "independent", "correlated"],
)
def test_seeded_four_chain_means_covariances_and_batch_mcse(
    mean, covariance, record_property
):
    started = time.perf_counter()
    chain = BlackJAXChain(
        gaussian_value_and_grad(mean, covariance),
        HMCControls(
            initial_step_size=0.3,
            adapt_step_size=False,
            step_size_bounds=(1e-6, np.pi / 8),
        ),
    )
    samples = np.empty((4, DRAWS, 3))
    divergent = 0
    for index, seed in enumerate(SEEDS):
        key = jax.random.key(seed)
        # Independent overdispersed starts and durations use a distinct NumPy
        # stream; all momentum and Metropolis randomness stays in BlackJAX.
        host_random = np.random.default_rng(seed + 100000)
        state = chain.initialize(
            mean + 2 * np.sqrt(np.diag(covariance)) * host_random.normal(size=3)
        )
        for draw in range(BURN + DRAWS):
            key, transition_key = jax.random.split(key)
            duration = host_random.uniform(*chain.controls.trajectory_interval)
            count = max(1, int(np.floor(duration / 0.3 + 0.5)))
            transition = chain.transition(
                state, key=transition_key, step_size=0.3, num_steps=count
            )
            state = transition.state
            divergent += transition.divergent
            if draw >= BURN:
                samples[index, draw - BURN] = state.position
    estimate, mcse = batch_estimates(samples, mean)
    target = np.concatenate([mean, covariance.ravel()])
    assert np.all(np.abs(estimate - target) <= 5 * mcse), (estimate, target, mcse)
    # Also check the usual sample covariance, rather than only exact-centered
    # second moments used to estimate its Monte Carlo uncertainty.
    assert np.all(
        np.abs(np.cov(samples.reshape(-1, 3), rowvar=False) - covariance)
        <= 5 * mcse[3:].reshape(3, 3)
    )
    assert np.all(np.isfinite(mcse)) and np.all(mcse > 0)
    scales = np.concatenate(
        [
            np.sqrt(np.diag(covariance)),
            np.sqrt(np.outer(np.diag(covariance), np.diag(covariance))).ravel(),
        ]
    )
    assert np.all(mcse / scales < 0.12), mcse / scales
    _, half_mcse = batch_estimates(samples[:, : DRAWS // 2], mean)
    # Doubling the draws must reduce the overall estimated uncertainty, with
    # finite-batch allowance around the asymptotic 1/sqrt(2) scaling.
    assert 0.45 < np.linalg.norm(mcse) / np.linalg.norm(half_mcse) < 0.95
    assert divergent == 0
    record_property("seeds", str(SEEDS))
    record_property("burn_per_chain", BURN)
    record_property("draws_per_chain", DRAWS)
    record_property("batch_length", BATCH)
    record_property(
        "maximum_standardized_error", float(np.max(np.abs(estimate - target) / mcse))
    )
    record_property(
        "mcse_length_ratio", float(np.linalg.norm(mcse) / np.linalg.norm(half_mcse))
    )
    record_property("seconds", time.perf_counter() - started)
