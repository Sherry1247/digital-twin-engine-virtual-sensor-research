"""Relative Gaussian noise fault injection."""

from __future__ import annotations

import numpy as np


def inject_relative_gaussian_noise(
    sensor_values: np.ndarray,
    sigma: float,
    random_state: int | None = None,
) -> np.ndarray:
    """Inject multiplicative Gaussian noise as ``x * (1 + Normal(0, sigma))``."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=sigma, size=sensor_values.shape)
    return sensor_values * (1.0 + noise)
