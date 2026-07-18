"""Gaussian-noise fault injection from a clean signal baseline."""

from __future__ import annotations

import numpy as np


def inject_relative_gaussian_noise(
    sensor_values: np.ndarray,
    severity: float,
    random_state: int | None = None,
) -> np.ndarray:
    """Add zero-mean noise with per-sample standard deviation ``severity * y``."""
    if severity < 0:
        raise ValueError("severity must be non-negative")

    values = np.asarray(sensor_values, dtype=float)
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=severity * np.abs(values), size=values.shape)
    return values + noise
