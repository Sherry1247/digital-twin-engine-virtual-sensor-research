"""Healthy sensor-signal generation for synthetic fault datasets."""

from __future__ import annotations

import numpy as np
def generate_healthy_signal(
    representative_value: float,
    n_samples: int,
    relative_noise: float,
    random_state: int | None,
) -> np.ndarray:
    """Create normal 2% measurement variation around a representative value."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if relative_noise < 0:
        raise ValueError("relative_noise must be non-negative")

    rng = np.random.default_rng(random_state)
    baseline = float(representative_value)
    measurement_noise = rng.normal(0.0, relative_noise * abs(baseline), size=n_samples)
    return baseline + measurement_noise
