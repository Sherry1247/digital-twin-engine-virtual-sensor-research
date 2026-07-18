"""Severity-scaled sensor lag fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_lag(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Delay a clean sensor signal by ``int(severity * 100)`` samples."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    if values.size == 0:
        return values.copy()

    delay = min(values.size - 1, int(severity * 100))
    if delay == 0:
        return values.copy()
    return np.concatenate((np.repeat(values[0], delay), values[:-delay]))
